import streamlit as st
import os
import pandas as pd
import faiss
import requests
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================================================
# Firebase Initialization
# =========================================================
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()
FIREBASE_API_KEY = st.secrets["FIREBASE_WEB_API_KEY"]

# =========================================================
# Firebase Auth
# =========================================================
def firebase_register(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    return requests.post(url, json={
        "email": email,
        "password": password,
        "returnSecureToken": True
    }).json()

def firebase_login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    return requests.post(url, json={
        "email": email,
        "password": password,
        "returnSecureToken": True
    }).json()

# =========================================================
# App UI
# =========================================================
st.set_page_config(page_title="MedCheck", layout="centered")

st.title("🩺 MedCheck")
st.subheader("Medical Claim Verification System")

st.markdown("""
MedCheck verifies medical advice using  
**evidence-based retrieval and controlled AI reasoning**.

⚠️ Informational use only. Not a substitute for professional medical advice.
""")

# =========================================================
# Session State
# =========================================================
for key in ["logged_in", "user_uid", "user_profile"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "logged_in" else False

# =========================================================
# AUTH
# =========================================================
st.markdown("## 🔐 Account")
mode = st.radio("Choose action", ["Login", "Register"], horizontal=True)

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if mode == "Register":
    age = st.number_input("Age", 0, 120)
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    conditions = st.text_input("Known medical conditions (comma-separated)")
    allergies = st.text_input("Allergies (comma-separated)")
    pregnancy = st.checkbox("Pregnant (if applicable)")

if st.button(mode):
    if mode == "Register":
        res = firebase_register(email, password)
        if "localId" in res:
            uid = res["localId"]
            db.collection("users").document(uid).set({
                "email": email,
                "age": age,
                "gender": gender,
                "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
                "allergies": [],
                "pregnancy_status": pregnancy
            })
            st.success("Registered! Please login.")
        else:
            st.error("Registration failed.")
    else:
        res = firebase_login(email, password)
        if "localId" in res:
            uid = res["localId"]
            doc = db.collection("users").document(uid).get()
            st.session_state.logged_in = True
            st.session_state.user_uid = uid
            st.session_state.user_profile = doc.to_dict()
            st.success("Logged in!")

if not st.session_state.logged_in:
    st.stop()

# =========================================================
# Profile Edit
# =========================================================
st.sidebar.markdown("## 👤 Profile Settings")
profile = st.session_state.user_profile

with st.sidebar.form("edit_profile"):
    age = st.number_input("Age", 0, 120, value=profile.get("age", 0))
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    conditions = st.text_input("Known medical conditions", ", ".join(profile.get("known_conditions", [])))
    pregnancy = st.checkbox("Pregnant", value=profile.get("pregnancy_status", False))
    save = st.form_submit_button("Save")

if save:
    updated = {
        "email": profile["email"],
        "age": age,
        "gender": gender,
        "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
        "pregnancy_status": pregnancy
    }
    db.collection("users").document(st.session_state.user_uid).set(updated)
    st.session_state.user_profile = updated
    st.sidebar.success("Profile updated")

# =========================================================
# Load FAISS
# =========================================================
@st.cache_resource
def load_resources():
    index = faiss.read_index("embeddings/faiss_index/medcheck.index")
    meta = pd.read_csv("embeddings/chunk_metadata.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, meta, embedder

index, metadata, embedder = load_resources()

def retrieve_medical_evidence(query, k=5):
    q = embedder.encode([query])
    _, idx = index.search(q, k)
    return [{"text": metadata.iloc[i]["text"]} for i in idx[0]]

# =========================================================
# Entity-aware Filtering
# =========================================================
def filter_evidence(query, evidence):
    stop = {"is","it","safe","to","for","a","the","of","and","or","once","day","days","every"}
    terms = {w for w in query.lower().split() if w not in stop}
    filtered = [e for e in evidence if any(t in e["text"].lower() for t in terms)]
    return filtered if filtered else evidence

# =========================================================
# LLM
# =========================================================
@st.cache_resource
def load_llm():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, mdl

llm_tokenizer, llm_model = load_llm()

# =========================================================
# Core Logic (STABLE v4)
# =========================================================
def classify_duration(c):
    c = c.lower()
    if any(x in c for x in ["2 days","few days","once"]): return "short"
    if any(x in c for x in ["months","every day","daily"]): return "long"
    return "unknown"

def extract_signals(evidence):
    text = " ".join(e["text"].lower() for e in evidence)
    return {
        "warns_normal": any(x in text for x in ["not safe","do not use","contraindicated"]),
        "warns_excessive": any(x in text for x in ["long-term","excessive","toxicity","risk"])
    }

def decide_verdict(duration, s):
    if s["warns_normal"]: return "Unsafe"
    if duration == "long" and s["warns_excessive"]: return "Unsafe"
    return "Safe"

def generate_explanation(claim, evidence, verdict):
    ev = "\n".join(f"- {e['text']}" for e in evidence)
    prompt = f"""
Explain the verdict "{verdict}" using ONLY the evidence.

Claim:
{claim}

Evidence:
{ev}

Explanation:
"""
    out = llm_model.generate(**llm_tokenizer(prompt, return_tensors="pt"), max_new_tokens=80)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True)

def sanitize(exp, evidence):
    bad = ["opioid","fda","children","pregnancy"]
    ev = " ".join(e["text"].lower() for e in evidence)
    if any(b in exp.lower() and b not in ev for b in bad):
        return "Based on the available medical evidence, no specific additional risks were identified."
    return exp

def verify_claim(claim, evidence):
    duration = classify_duration(claim)
    signals = extract_signals(evidence)
    verdict = decide_verdict(duration, signals)
    exp = sanitize(generate_explanation(claim, evidence, verdict), evidence)
    return verdict, exp

# =========================================================
# UI – Verification
# =========================================================
st.markdown("## 🧠 Verify Medical Claim")
claim = st.text_area("Enter a medical claim")

if st.button("Verify"):
    evidence = filter_evidence(claim, retrieve_medical_evidence(claim))
    verdict, explanation = verify_claim(claim, evidence)

    st.markdown("## 🧾 Verdict")
    if verdict == "Safe":
        st.success(verdict)
    else:
        st.error(verdict)

    st.markdown("## 🔍 Explanation")
    st.write(explanation)

    st.markdown("## ⚠️ Personalized Safety Assessment")
    warnings = []
    text = " ".join(e["text"].lower() for e in evidence)

    for c in st.session_state.user_profile.get("known_conditions", []):
        if c in text:
            warnings.append(f"You have {c}. Evidence mentions related risks.")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.info("No additional personalized risks identified.")

    st.markdown("## 📚 Supporting Medical Evidence")
    for e in evidence:
        st.markdown(f"- {e['text']}")

st.markdown("---")
st.caption("⚠️ MedCheck v4.3 • Informational use only")
