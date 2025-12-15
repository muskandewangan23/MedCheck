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
# Firebase Auth (REST)
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
for k in ["logged_in", "user_profile", "user_uid"]:
    if k not in st.session_state:
        st.session_state[k] = None

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
            db.collection("users").document(res["localId"]).set({
                "email": email,
                "age": age,
                "gender": gender,
                "known_conditions": [c.strip().lower() for c in conditions.split(",") if c],
                "allergies": [a.strip().lower() for a in allergies.split(",") if a],
                "pregnancy_status": pregnancy
            })
            st.success("Registered successfully. Please login.")
        else:
            st.error(res)
    else:
        res = firebase_login(email, password)
        if "localId" in res:
            doc = db.collection("users").document(res["localId"]).get()
            st.session_state.logged_in = True
            st.session_state.user_uid = res["localId"]
            st.session_state.user_profile = doc.to_dict()
            st.success("Logged in successfully.")
        else:
            st.error(res)

if not st.session_state.logged_in:
    st.stop()

# =========================================================
# PROFILE EDIT
# =========================================================
st.sidebar.markdown("## 👤 Profile Settings")
p = st.session_state.user_profile

with st.sidebar.form("edit_profile"):
    age = st.number_input("Age", 0, 120, value=p.get("age", 0))
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"],
                          index=["Prefer not to say", "Male", "Female", "Other"].index(p.get("gender")))
    conditions = st.text_input("Known medical conditions", ", ".join(p.get("known_conditions", [])))
    allergies = st.text_input("Allergies", ", ".join(p.get("allergies", [])))
    pregnancy = st.checkbox("Pregnant", value=p.get("pregnancy_status", False))

    if st.form_submit_button("Save"):
        updated = {
            "email": p["email"],
            "age": age,
            "gender": gender,
            "known_conditions": [c.strip().lower() for c in conditions.split(",") if c],
            "allergies": [a.strip().lower() for a in allergies.split(",") if a],
            "pregnancy_status": pregnancy
        }
        db.collection("users").document(st.session_state.user_uid).set(updated)
        st.session_state.user_profile = updated
        st.sidebar.success("Profile updated")

# =========================================================
# Load FAISS + Models
# =========================================================
@st.cache_resource
def load_resources():
    index = faiss.read_index("embeddings/faiss_index/medcheck.index")
    meta = pd.read_csv("embeddings/chunk_metadata.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return index, meta, embedder, tokenizer, model

index, meta, embedder, llm_tokenizer, llm_model = load_resources()

def retrieve_evidence(q, k=5):
    emb = embedder.encode([q])
    _, idx = index.search(emb, k)
    return [{"text": meta.iloc[i]["text"]} for i in idx[0]]

def filter_entity(q, evidence):
    terms = q.lower().split()
    out = [e for e in evidence if any(t in e["text"].lower() for t in terms)]
    return out if out else evidence

# =========================================================
# 🔒 LOCKED v4 LOGIC
# =========================================================
def is_non_medical(claim):
    return any(x in claim.lower() for x in ["milk", "water", "lemon", "tea", "coffee"])

def has_direct_contradiction(evidence):
    t = " ".join(e["text"].lower() for e in evidence)
    return "do not treat viral" in t or "does not treat viral" in t

def classify_duration(claim):
    c = claim.lower()
    if any(x in c for x in ["2 days", "few days", "once"]):
        return "short"
    if any(x in c for x in ["months", "every day", "daily"]):
        return "long"
    return "unknown"

def decide_verdict(duration, evidence):
    t = " ".join(e["text"].lower() for e in evidence)

    if "should not be used" in t or "contraindicated" in t:
        return "Unsafe"

    if duration == "long" and any(x in t for x in ["long-term", "toxicity", "risk of"]):
        return "Unsafe"

    return "Safe"

def generate_explanation(claim, evidence, verdict):
    ev = "\n".join("- " + e["text"] for e in evidence)
    prompt = f"""
Use ONLY the evidence below.
Do NOT add new facts.

Claim:
{claim}

Evidence:
{ev}

Explain why verdict is {verdict} in 2 sentences.
"""
    out = llm_model.generate(**llm_tokenizer(prompt, return_tensors="pt"), max_new_tokens=80)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True)

# =========================================================
# UI – VERIFY
# =========================================================
st.markdown("## 🧠 Verify Medical Claim")
claim = st.text_area("Enter a medical claim")

if st.button("Verify"):
    evidence = filter_entity(claim, retrieve_evidence(claim))

    if is_non_medical(claim):
        verdict, explanation = "Safe", "No medical safety risks are indicated for normal consumption."
    elif has_direct_contradiction(evidence):
        verdict = "Unsafe"
        explanation = generate_explanation(claim, evidence, verdict)
    else:
        verdict = decide_verdict(classify_duration(claim), evidence)
        explanation = generate_explanation(claim, evidence, verdict)

    st.markdown("## 🧾 Verdict")
    st.success(verdict) if verdict == "Safe" else st.error(verdict)

    st.markdown("## 🔍 Explanation")
    st.write(explanation)

    # Personalized
    st.markdown("## ⚠️ Personalized Safety Assessment")
    txt = " ".join(e["text"].lower() for e in evidence)
    warnings = []

    for c in st.session_state.user_profile.get("known_conditions", []):
        if c in txt:
            warnings.append(f"Your condition ({c}) is mentioned in the evidence.")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.info("No additional personalized risks identified.")

    st.markdown("## 📚 Supporting Medical Evidence")
    for e in evidence:
        st.markdown(f"- {e['text']}")

st.caption("⚠️ MedCheck v4.3 • Informational use only.")
