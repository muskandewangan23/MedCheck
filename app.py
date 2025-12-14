import streamlit as st
import os
import numpy as np
import pandas as pd
import faiss
import requests
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, firestore

# =============================
# Firebase Initialization
# =============================
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        firebase_secrets = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_secrets)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()
FIREBASE_API_KEY = st.secrets["FIREBASE_WEB_API_KEY"]

# =============================
# Firebase Auth (REST)
# =============================
def firebase_register(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return requests.post(url, json=payload).json()

def firebase_login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return requests.post(url, json=payload).json()

# =============================
# App Configuration
# =============================
st.set_page_config(page_title="MedCheck", layout="centered")

st.title("🩺 MedCheck")
st.subheader("Medical Claim Verification System")

st.markdown("""
MedCheck verifies medical advice or health-related claims using  
**evidence-based retrieval and reasoning**.

⚠️ This system provides **informational guidance only** and is **not a substitute
for professional medical advice**.
""")

# =============================
# Session State
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "user_uid" not in st.session_state:
    st.session_state.user_uid = None

# =============================
# AUTH UI
# =============================
st.markdown("## 🔐 Account")

auth_mode = st.radio("Choose action", ["Login", "Register"], horizontal=True)

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if auth_mode == "Register":
    st.markdown("### 👤 Medical Profile (Stored once)")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    conditions = st.text_input("Known medical conditions (comma-separated)")
    allergies = st.text_input("Allergies (comma-separated)")
    pregnancy = st.checkbox("Pregnant (if applicable)")

if st.button(auth_mode):
    if not email or not password:
        st.error("Email and password are required.")
    else:
        if auth_mode == "Register":
            auth_res = firebase_register(email, password)

            if "localId" in auth_res:
                uid = auth_res["localId"]
                profile = {
                    "email": email,
                    "age": age,
                    "gender": gender,
                    "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
                    "allergies": [a.strip().lower() for a in allergies.split(",") if a.strip()],
                    "pregnancy_status": pregnancy
                }
                db.collection("users").document(uid).set(profile)
                st.success("Registration successful! Please login.")
            else:
                st.error(auth_res.get("error", {}).get("message", "Registration failed"))

        else:  # Login
            auth_res = firebase_login(email, password)

            if "localId" in auth_res:
                uid = auth_res["localId"]
                doc = db.collection("users").document(uid).get()

                if doc.exists:
                    st.session_state.logged_in = True
                    st.session_state.user_profile = doc.to_dict()
                    st.session_state.user_uid = uid
                    st.success("Logged in successfully!")
                else:
                    st.error("User profile not found.")
            else:
                st.error(auth_res.get("error", {}).get("message", "Login failed"))

# =============================
# STOP if not logged in
# =============================
if not st.session_state.logged_in:
    st.info("Please login or register to use MedCheck.")
    st.stop()

# =============================
# EDIT PROFILE (v3.1)
# =============================
st.sidebar.markdown("## 👤 Profile Settings")
profile = st.session_state.user_profile

with st.sidebar.form("edit_profile_form"):
    age = st.number_input("Age", 0, 120, value=profile.get("age") or 0)
    gender = st.selectbox(
        "Gender",
        ["Prefer not to say", "Male", "Female", "Other"],
        index=["Prefer not to say", "Male", "Female", "Other"].index(
            profile.get("gender", "Prefer not to say")
        )
    )
    conditions = st.text_input(
        "Known medical conditions (comma-separated)",
        value=", ".join(profile.get("known_conditions", []))
    )
    allergies = st.text_input(
        "Allergies (comma-separated)",
        value=", ".join(profile.get("allergies", []))
    )
    pregnancy = st.checkbox(
        "Pregnant (if applicable)",
        value=profile.get("pregnancy_status", False)
    )

    save_profile = st.form_submit_button("💾 Save Profile")

if save_profile:
    updated_profile = {
        "email": profile.get("email"),
        "age": age,
        "gender": gender,
        "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
        "allergies": [a.strip().lower() for a in allergies.split(",") if a.strip()],
        "pregnancy_status": pregnancy
    }

    db.collection("users").document(st.session_state.user_uid).set(updated_profile)
    st.session_state.user_profile = updated_profile
    st.sidebar.success("Profile updated successfully!")

# =============================
# Load Resources (v1)
# =============================
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

# =============================
# Retriever Agent
# =============================
def retrieve_medical_evidence(query, top_k=3):
    q_emb = embedder.encode([query])
    distances, indices = index.search(q_emb, top_k)
    return [{
        "text": metadata.iloc[idx]["text"],
        "distance": float(distances[0][i])
    } for i, idx in enumerate(indices[0])]

# =============================
# Verification Agent
# =============================
def verify_medical_claim(claim, evidence):
    combined = " ".join(e["text"].lower() for e in evidence)
    if "long-term" in combined or "excessive" in combined:
        return {"verdict": "Unsafe", "explanation": "Long-term or excessive use may cause health risks."}
    return {"verdict": "Partially Accurate", "explanation": "The claim is not fully supported by evidence."}

# =============================
# Personalization Agent (v2)
# =============================
def personalize_risk(profile, evidence):
    warnings = []
    evidence_text = " ".join(e["text"].lower() for e in evidence)

    for c in profile.get("known_conditions", []):
        if "kidney" in c and "kidney" in evidence_text:
            warnings.append("Your kidney condition may increase risk.")
        if "liver" in c and "liver" in evidence_text:
            warnings.append("Your liver condition may increase risk.")
    return warnings

# =============================
# Confidence
# =============================
def compute_confidence(verdict, k):
    base = {"Accurate": 0.8, "Partially Accurate": 0.65, "Unsafe": 0.85}
    return round(min(base.get(verdict, 0.6) + 0.01 * k, 0.95), 2)

# =============================
# Claim UI
# =============================
st.markdown("## 🧠 Verify Medical Claim")

claim = st.text_area("Enter a medical claim or advice", height=120)

if st.button("Verify Claim"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    else:
        evidence = retrieve_medical_evidence(claim, top_k=2)
        verification = verify_medical_claim(claim, evidence)
        confidence = compute_confidence(verification["verdict"], len(evidence))

        st.markdown("## 🧾 Verdict")
        if verification["verdict"] == "Unsafe":
            st.error(verification["verdict"])
        else:
            st.success(verification["verdict"])

        st.markdown("## 🔍 Explanation")
        st.write(verification["explanation"])

        st.markdown(f"## 📊 Confidence Score: `{confidence}`")

        warnings = personalize_risk(st.session_state.user_profile, evidence)

        st.markdown("## ⚠️ Personalized Safety Assessment")
        
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.info(
                "Based on your profile, no specific additional risks were identified "
                "beyond the general medical evidence."
            )

        st.markdown("## 📚 Supporting Medical Evidence")
        for e in evidence:
            st.markdown(f"- {e['text']}")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("⚠️ MedCheck v3.1 • Informational use only. Consult a medical professional.")

