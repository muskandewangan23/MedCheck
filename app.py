import streamlit as st
import os
import numpy as np
import pandas as pd
import faiss
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
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# =============================
# AUTH UI (v3 Step 5)
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
            profile = {
                "email": email,
                "age": age,
                "gender": gender,
                "known_conditions": [c.strip().lower() for c in conditions.split(",") if c],
                "allergies": [a.strip().lower() for a in allergies.split(",") if a],
                "pregnancy_status": pregnancy
            }
            db.collection("users").document(email).set(profile)
            st.success("Registration successful! You can now login.")

        else:  # Login
            doc = db.collection("users").document(email).get()
            if doc.exists:
                st.session_state.logged_in = True
                st.session_state.user_profile = doc.to_dict()
                st.session_state.user_email = email
                st.success("Logged in successfully!")
            else:
                st.error("User not found. Please register first.")

# =============================
# STOP if not logged in
# =============================
if not st.session_state.logged_in:
    st.info("Please login or register to use MedCheck.")
    st.stop()

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
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    return [{
        "rank": i + 1,
        "text": metadata.iloc[idx]["text"],
        "distance": float(distances[0][i])
    } for i, idx in enumerate(indices[0])]

# =============================
# Verification Agent
# =============================
def verify_medical_claim(claim, evidence):
    combined = " ".join(e["text"].lower() for e in evidence)
    if "long-term" in combined or "excessive" in combined:
        return {
            "verdict": "Unsafe",
            "explanation": "Long-term or excessive use is associated with serious health risks."
        }
    return {
        "verdict": "Partially Accurate",
        "explanation": "The claim is not fully supported by the evidence."
    }

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

claim = st.text_area(
    "Enter a medical claim or advice",
    height=120,
    placeholder="Example: It is safe to take ibuprofen every day for a long time."
)

if st.button("Verify Claim"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    else:
        evidence = retrieve_medical_evidence(claim, top_k=2)
        verification = verify_medical_claim(claim, evidence)
        confidence = compute_confidence(verification["verdict"], len(evidence))

        st.markdown("## 🧾 Verdict")
        st.error(verification["verdict"]) if verification["verdict"] == "Unsafe" else st.success(verification["verdict"])

        st.markdown("## 🔍 Explanation")
        st.write(verification["explanation"])

        st.markdown(f"## 📊 Confidence Score: `{confidence}`")

        warnings = personalize_risk(st.session_state.user_profile, evidence)
        if warnings:
            st.markdown("## ⚠️ Personalized Safety Warnings")
            for w in warnings:
                st.warning(w)

        st.markdown("## 📚 Supporting Medical Evidence")
        for e in evidence:
            st.markdown(f"- {e['text']}")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("⚠️ MedCheck v3 • Informational use only. Consult a medical professional.")
