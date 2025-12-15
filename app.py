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
for key in ["logged_in", "user_profile", "user_uid"]:
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
    conditions = st.text_input("Known medical conditions")
    pregnancy = st.checkbox("Pregnant (if applicable)")

if st.button(mode):
    if mode == "Register":
        res = firebase_register(email, password)
        if "localId" in res:
            db.collection("users").document(res["localId"]).set({
                "email": email,
                "age": age,
                "gender": gender,
                "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
                "pregnancy_status": pregnancy
            })
            st.success("Registered. Please login.")
    else:
        res = firebase_login(email, password)
        if "localId" in res:
            doc = db.collection("users").document(res["localId"]).get()
            if doc.exists:
                st.session_state.logged_in = True
                st.session_state.user_uid = res["localId"]
                st.session_state.user_profile = doc.to_dict()
                st.success("Logged in")

if not st.session_state.logged_in:
    st.stop()

# =========================================================
# PROFILE EDIT
# =========================================================
st.sidebar.markdown("## 👤 Profile Settings")
p = st.session_state.user_profile

with st.sidebar.form("profile"):
    age = st.number_input("Age", 0, 120, value=p.get("age", 0))
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"],
                          index=["Prefer not to say", "Male", "Female", "Other"].index(p.get("gender", "Prefer not to say")))
    conditions = st.text_input("Known medical conditions", ", ".join(p.get("known_conditions", [])))
    pregnancy = st.checkbox("Pregnant", value=p.get("pregnancy_status", False))
    if st.form_submit_button("Save"):
        updated = {
            "email": p["email"],
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
def load_retriever():
    index = faiss.read_index("embeddings/faiss_index/medcheck.index")
    meta = pd.read_csv("embeddings/chunk_metadata.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, meta, embedder

index, metadata, embedder = load_retriever()

def retrieve_evidence(query, k=5):
    q_emb = embedder.encode([query])
    _, idxs = index.search(q_emb, k)

    q_terms = set(query.lower().split())
    filtered = []

    for i in idxs[0]:
        text = metadata.iloc[i]["text"]
        if any(t in text.lower() for t in q_terms):
            filtered.append({"text": text})

    return filtered[:2] if filtered else [{"text": metadata.iloc[idxs[0][0]]["text"]}]

# =========================================================
# LLM (Explanation Only)
# =========================================================
@st.cache_resource
def load_llm():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, mdl

tok, mdl = load_llm()

def classify_duration(claim):
    c = claim.lower()
    if any(x in c for x in ["1 day", "2 days", "few days", "once"]):
        return "short"
    if any(x in c for x in ["months", "weeks", "daily", "every day"]):
        return "long"
    return "unknown"

def decide_verdict(duration, evidence):
    text = " ".join(e["text"].lower() for e in evidence)

    if any(p in text for p in ["do not use", "not safe", "contraindicated"]):
        return "Unsafe"

    if duration == "long" and any(p in text for p in ["long-term", "excessive", "toxicity", "risk"]):
        return "Unsafe"

    return "Safe"

def generate_explanation(claim, evidence, verdict):
    ev = "\n".join(f"- {e['text']}" for e in evidence)
    prompt = f"""
Explain why the verdict is "{verdict}" using ONLY the evidence.

Claim:
{claim}

Evidence:
{ev}

Explanation:
"""
    inputs = tok(prompt, return_tensors="pt", truncation=True)
    out = mdl.generate(**inputs, max_new_tokens=80)
    return tok.decode(out[0], skip_special_tokens=True)

# =========================================================
# UI – CLAIM
# =========================================================
st.markdown("## 🧠 Verify Medical Claim")
claim = st.text_area("Enter a medical claim")

if st.button("Verify"):
    evidence = retrieve_evidence(claim)
    duration = classify_duration(claim)
    verdict = decide_verdict(duration, evidence)
    explanation = generate_explanation(claim, evidence, verdict)

    st.markdown("## 🧾 Verdict")
    if verdict == "Safe":
        st.success(verdict)
    else:
        st.error(verdict)

    st.markdown("## 🔍 Explanation")
    st.write(explanation)

    # Personalization
    st.markdown("## ⚠️ Personalized Safety Assessment")
    profile = st.session_state.user_profile
    warnings = []
    ev_text = " ".join(e["text"].lower() for e in evidence)

    for c in profile.get("known_conditions", []):
        if "kidney" in c and "kidney" in ev_text:
            warnings.append("Kidney-related condition may increase risk.")
        if "liver" in c and "liver" in ev_text:
            warnings.append("Liver-related condition may increase risk.")

    if profile.get("pregnancy_status"):
        warnings.append("Pregnancy may introduce additional risks.")

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
