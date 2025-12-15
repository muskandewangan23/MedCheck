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
        firebase_secrets = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_secrets)
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
# App Config
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
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "user_uid" not in st.session_state:
    st.session_state.user_uid = None

# =========================================================
# AUTH UI
# =========================================================
st.markdown("## 🔐 Account")
auth_mode = st.radio("Choose action", ["Login", "Register"], horizontal=True)

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if auth_mode == "Register":
    st.markdown("### 👤 Medical Profile (Stored once)")
    age = st.number_input("Age", 0, 120)
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    conditions = st.text_input("Known medical conditions (comma-separated)")
    pregnancy = st.checkbox("Pregnant (if applicable)")

if st.button(auth_mode):
    if auth_mode == "Register":
        res = firebase_register(email, password)
        if "localId" in res:
            uid = res["localId"]
            db.collection("users").document(uid).set({
                "email": email,
                "age": age,
                "gender": gender,
                "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
                "pregnancy_status": pregnancy
            })
            st.success("Registration successful! Please login.")
    else:
        res = firebase_login(email, password)
        if "localId" in res:
            uid = res["localId"]
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                st.session_state.logged_in = True
                st.session_state.user_uid = uid
                st.session_state.user_profile = doc.to_dict()
                st.success("Logged in successfully!")

if not st.session_state.logged_in:
    st.stop()

# =========================================================
# Profile Edit (VISIBLE & WORKING)
# =========================================================
st.sidebar.markdown("## 👤 Profile Settings")
profile = st.session_state.user_profile

with st.sidebar.form("profile"):
    age = st.number_input("Age", value=profile["age"])
    gender = st.selectbox(
        "Gender",
        ["Prefer not to say", "Male", "Female", "Other"],
        index=["Prefer not to say", "Male", "Female", "Other"].index(profile["gender"])
    )
    conditions = st.text_input("Known medical conditions", ", ".join(profile["known_conditions"]))
    save = st.form_submit_button("Save")

if save:
    profile["age"] = age
    profile["gender"] = gender
    profile["known_conditions"] = [c.strip().lower() for c in conditions.split(",") if c.strip()]
    db.collection("users").document(st.session_state.user_uid).set(profile)
    st.sidebar.success("Profile updated")

# =========================================================
# Load FAISS + Embeddings
# =========================================================
BASE_DIR = "."
FAISS_PATH = os.path.join(BASE_DIR, "embeddings/faiss_index/medcheck.index")
META_PATH = os.path.join(BASE_DIR, "embeddings/chunk_metadata.csv")

@st.cache_resource
def load_resources():
    index = faiss.read_index(FAISS_PATH)
    metadata = pd.read_csv(META_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, embedder

index, metadata, embedder = load_resources()

def retrieve_medical_evidence(query, top_k=5):
    emb = embedder.encode([query])
    _, idx = index.search(emb, top_k)
    return [{"text": metadata.iloc[i]["text"]} for i in idx[0]]

def filter_evidence(query, evidence):
    q = query.lower().split()
    filtered = [e for e in evidence if any(t in e["text"].lower() for t in q)]
    return filtered if filtered else evidence[:1]

# =========================================================
# 🔒 Locked v4.3 Reasoning Logic
# =========================================================
def classify_duration(claim):
    c = claim.lower()
    if any(x in c for x in ["2 days", "few days", "once"]):
        return "short"
    if any(x in c for x in ["months", "weeks", "daily", "every day"]):
        return "long"
    return "unknown"

def extract_evidence_signals(evidence):
    text = " ".join(e["text"].lower() for e in evidence)
    return {
        "warns_normal": any(p in text for p in ["do not treat", "not safe", "contraindicated"]),
        "warns_excessive": any(p in text for p in ["long-term", "excessive", "toxicity", "risk"])
    }

def decide_verdict(duration, signals):
    if signals["warns_normal"]:
        return "Unsafe"
    if duration == "long" and signals["warns_excessive"]:
        return "Unsafe"
    return "Safe"

# =========================================================
# Explanation (STRICT & SAFE)
# =========================================================
@st.cache_resource
def load_llm():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, mdl

llm_tokenizer, llm_model = load_llm()

def generate_explanation(claim, evidence, verdict):
    ev = "\n".join(f"- {e['text']}" for e in evidence)
    prompt = f"""
Explain the verdict "{verdict}" using ONLY the evidence.
Do not introduce new medical facts.

Claim:
{claim}

Evidence:
{ev}

Explanation:
"""
    inp = llm_tokenizer(prompt, return_tensors="pt", truncation=True)
    out = llm_model.generate(**inp, max_new_tokens=80)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True)

# =========================================================
# UI – Claim Verification
# =========================================================
st.markdown("## 🧠 Verify Medical Claim")
claim = st.text_area("Enter a medical claim")

if st.button("Verify Claim"):
    raw = retrieve_medical_evidence(claim)
    evidence = filter_evidence(claim, raw)

    duration = classify_duration(claim)
    signals = extract_evidence_signals(evidence)
    verdict = decide_verdict(duration, signals)
    explanation = generate_explanation(claim, evidence, verdict)

    st.markdown("## 🧾 Verdict")
    if verdict == "Unsafe":
        st.error(verdict)
    else:
        st.success(verdict)

    st.markdown("## 🔍 Explanation")
    st.write(explanation)

    st.markdown("## ⚠️ Personalized Safety Assessment")
    ev_text = " ".join(e["text"].lower() for e in evidence)
    warned = False
    for c in profile["known_conditions"]:
        if c in ev_text:
            st.warning(f"You have {c}. The evidence mentions related risks.")
            warned = True
    if not warned:
        st.info("No additional personalized risks identified.")

    st.markdown("## 📚 Supporting Medical Evidence")
    for e in evidence:
        st.markdown(f"- {e['text']}")

st.markdown("---")
st.caption("⚠️ MedCheck v4.3 • Informational use only.")
