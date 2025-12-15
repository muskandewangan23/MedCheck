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
    return requests.post(url, json={"email": email, "password": password, "returnSecureToken": True}).json()

def firebase_login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    return requests.post(url, json={"email": email, "password": password, "returnSecureToken": True}).json()

# =========================================================
# App Config
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
for key in ["logged_in", "user_profile", "user_uid"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "logged_in" else False

# =========================================================
# AUTH UI
# =========================================================
st.markdown("## 🔐 Account")
auth_mode = st.radio("Choose action", ["Login", "Register"], horizontal=True)
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if auth_mode == "Register":
    age = st.number_input("Age", 0, 120)
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
            st.success("Registered successfully. Please login.")
    else:
        res = firebase_login(email, password)
        if "localId" in res:
            uid = res["localId"]
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                st.session_state.logged_in = True
                st.session_state.user_uid = uid
                st.session_state.user_profile = doc.to_dict()

if not st.session_state.logged_in:
    st.stop()

# =========================================================
# Load FAISS
# =========================================================
@st.cache_resource
def load_resources():
    index = faiss.read_index("embeddings/faiss_index/medcheck.index")
    metadata = pd.read_csv("embeddings/chunk_metadata.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, embedder

index, metadata, embedder = load_resources()

def retrieve_medical_evidence(query, k=5):
    emb = embedder.encode([query])
    _, idxs = index.search(emb, k)
    return [{"text": metadata.iloc[i]["text"]} for i in idxs[0]]

def filter_evidence(query, evidence):
    q = set(query.lower().split())
    filtered = [e for e in evidence if any(w in e["text"].lower() for w in q)]
    return filtered or evidence

# =========================================================
# LLM (Explanation only)
# =========================================================
@st.cache_resource
def load_llm():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, mdl

llm_tokenizer, llm_model = load_llm()

# =========================================================
# 🔒 FINAL STABLE v4.0 LOGIC
# =========================================================
def classify_duration(claim):
    c = claim.lower()
    if any(x in c for x in ["2 days", "3 days", "few days", "once"]):
        return "short"
    if any(x in c for x in ["months", "weeks", "every day", "daily"]):
        return "long"
    return "unknown"

def extract_risk_signals(evidence):
    text = " ".join(e["text"].lower() for e in evidence)
    return {
        "warns_normal": any(p in text for p in ["not safe", "do not use", "does not treat", "contraindicated"]),
        "warns_long": any(p in text for p in ["long-term", "excessive", "toxicity", "organ damage", "risk"])
    }

def decide_verdict_v4(duration, signals):
    if signals["warns_normal"]:
        return "Unsafe"
    if duration == "long" and signals["warns_long"]:
        return "Unsafe"
    return "Safe"

def generate_explanation(claim, evidence, verdict):
    ev = "\n".join(f"- {e['text']}" for e in evidence)
    prompt = f"""
Explain the verdict below using ONLY the medical evidence.

Verdict: {verdict}
Claim: {claim}
Evidence:
{ev}

Explanation (2–3 sentences):
"""
    inp = llm_tokenizer(prompt, return_tensors="pt", truncation=True)
    out = llm_model.generate(**inp, max_new_tokens=100)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True)

def verify_medical_claim_llm(claim, evidence):
    duration = classify_duration(claim)
    signals = extract_risk_signals(evidence)
    verdict = decide_verdict_v4(duration, signals)
    explanation = generate_explanation(claim, evidence, verdict)
    return verdict, explanation

# =========================================================
# UI
# =========================================================
st.markdown("## 🧠 Verify Medical Claim")
claim = st.text_area("Enter a medical claim")

if st.button("Verify Claim"):
    raw = retrieve_medical_evidence(claim)
    evidence = filter_evidence(claim, raw)

    verdict, explanation = verify_medical_claim_llm(claim, evidence)

    st.markdown("## 🧾 Verdict")
    st.success(verdict) if verdict == "Safe" else st.error(verdict)

    st.markdown("## 🔍 Explanation")
    st.write(explanation)

    # =============================
    # Personalized Safety
    # =============================
    st.markdown("## ⚠️ Personalized Safety Assessment")
    profile = st.session_state.user_profile
    ev_text = " ".join(e["text"].lower() for e in evidence)

    warnings = []
    if "kidney" in ev_text and "kidney" in " ".join(profile.get("known_conditions", [])):
        warnings.append("You have a kidney-related condition. The evidence mentions kidney risks.")
    if "liver" in ev_text and "liver" in " ".join(profile.get("known_conditions", [])):
        warnings.append("You have a liver-related condition. The evidence mentions liver toxicity.")
    if profile.get("pregnancy_status"):
        warnings.append("You are pregnant. Some medications may carry additional risks.")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.info("No additional personal risk factors identified based on your profile.")

    st.markdown("## 📚 Supporting Medical Evidence")
    for e in evidence:
        st.markdown(f"- {e['text']}")

st.markdown("---")
st.caption("⚠️ MedCheck v4.3 • Informational use only")
