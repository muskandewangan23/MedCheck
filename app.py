import streamlit as st
import os
import numpy as np
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
# App Configuration
# =========================================================
st.set_page_config(page_title="MedCheck", layout="centered")

st.title("🩺 MedCheck")
st.subheader("Medical Claim Verification System")

st.markdown("""
MedCheck verifies medical advice or health-related claims using  
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
    age = st.number_input("Age", 0, 120, step=1)
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    conditions = st.text_input("Known medical conditions (comma-separated)")
    allergies = st.text_input("Allergies (comma-separated)")
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
                "allergies": [a.strip().lower() for a in allergies.split(",") if a.strip()],
                "pregnancy_status": pregnancy
            })
            st.success("Registration successful. Please login.")
        else:
            st.error(res.get("error", {}).get("message", "Registration failed"))
    else:
        res = firebase_login(email, password)
        if "localId" in res:
            uid = res["localId"]
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                st.session_state.logged_in = True
                st.session_state.user_uid = uid
                st.session_state.user_profile = doc.to_dict()
                st.success("Logged in successfully.")
            else:
                st.error("User profile not found.")
        else:
            st.error(res.get("error", {}).get("message", "Login failed"))

if not st.session_state.logged_in:
    st.stop()

# =========================================================
# PROFILE EDIT
# =========================================================
st.sidebar.markdown("## 👤 Profile Settings")
profile = st.session_state.user_profile

with st.sidebar.form("edit_profile"):
    age = st.number_input("Age", 0, 120, value=profile.get("age", 0))
    gender = st.selectbox(
        "Gender",
        ["Prefer not to say", "Male", "Female", "Other"],
        index=["Prefer not to say", "Male", "Female", "Other"].index(profile.get("gender", "Prefer not to say"))
    )
    conditions = st.text_input("Known medical conditions", ", ".join(profile.get("known_conditions", [])))
    allergies = st.text_input("Allergies", ", ".join(profile.get("allergies", [])))
    pregnancy = st.checkbox("Pregnant", value=profile.get("pregnancy_status", False))
    save = st.form_submit_button("💾 Save Profile")

if save:
    updated = {
        "email": profile["email"],
        "age": age,
        "gender": gender,
        "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
        "allergies": [a.strip().lower() for a in allergies.split(",") if a.strip()],
        "pregnancy_status": pregnancy
    }
    db.collection("users").document(st.session_state.user_uid).set(updated)
    st.session_state.user_profile = updated
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
    _, idxs = index.search(emb, top_k)
    return [{"text": metadata.iloc[i]["text"]} for i in idxs[0]]

def filter_evidence_by_query_entities(query, evidence):
    terms = set(query.lower().split())
    filtered = [e for e in evidence if any(t in e["text"].lower() for t in terms)]
    return filtered if filtered else evidence

# =========================================================
# Load LLM
# =========================================================
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

llm_tokenizer, llm_model = load_llm()

# =========================================================
# Core Reasoning (LOCKED v4)
# =========================================================
def classify_duration(claim):
    c = claim.lower()
    if any(x in c for x in ["1 day", "2 days", "few days", "once", "short term"]):
        return "short"
    if any(x in c for x in ["weeks", "months", "every day", "daily", "long term"]):
        return "long"
    return "unknown"

def extract_evidence_signals(evidence):
    text = " ".join(e["text"].lower() for e in evidence)
    return {
        "warns_normal": any(p in text for p in ["not safe", "do not use", "do not treat", "contraindicated"]),
        "warns_excessive": any(p in text for p in ["long-term", "excessive", "toxicity", "organ damage", "risk"])
    }

def decide_verdict(duration, signals):
    if signals["warns_normal"]:
        return "Unsafe"
    if duration == "long" and signals["warns_excessive"]:
        return "Unsafe"
    return "Safe"

def generate_explanation(claim, evidence, verdict):
    ev = "\n".join(f"- {e['text']}" for e in evidence)
    prompt = f"""
Explain the verdict using ONLY the evidence.

Verdict: {verdict}
Claim: {claim}

Evidence:
{ev}

If Safe, explain why risks do not apply.
If Unsafe, explain the risk.

Explanation:
"""
    inp = llm_tokenizer(prompt, return_tensors="pt", truncation=True)
    out = llm_model.generate(**inp, max_new_tokens=120)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True)

def verify_medical_claim_llm(claim, evidence):
    duration = classify_duration(claim)
    signals = extract_evidence_signals(evidence)
    verdict = decide_verdict(duration, signals)
    explanation = generate_explanation(claim, evidence, verdict)
    return verdict, explanation

# =========================================================
# UI – Claim Verification
# =========================================================
st.markdown("## 🧠 Verify Medical Claim")
claim = st.text_area("Enter a medical claim", height=120)

if st.button("Verify Claim"):
    raw = retrieve_medical_evidence(claim)
    evidence = filter_evidence_by_query_entities(claim, raw)

    verdict, explanation = verify_medical_claim_llm(claim, evidence)

    st.markdown("## 🧾 Verdict")
    st.success(verdict) if verdict == "Safe" else st.error(verdict)

    st.markdown("## 🔍 Explanation")
    st.write(explanation)

    st.markdown("## ⚠️ Personalized Safety Assessment")
    profile = st.session_state.user_profile
    text = " ".join(e["text"].lower() for e in evidence)

    warnings = []
    for c in profile.get("known_conditions", []):
        if "kidney" in c and "kidney" in text:
            warnings.append("You have a kidney-related condition. The evidence mentions kidney risks.")
        if "liver" in c and "liver" in text:
            warnings.append("You have a liver-related condition. The evidence mentions liver toxicity.")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.info("No additional personalized risks identified based on your profile.")

    st.markdown("## 📚 Supporting Medical Evidence")
    for e in evidence:
        st.markdown(f"- {e['text']}")

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("⚠️ MedCheck v4.3 • Informational use only.")
