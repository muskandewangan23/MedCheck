import streamlit as st
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# =============================
# App Configuration
# =============================
st.set_page_config(
    page_title="MedCheck",
    layout="centered"
)

st.title("🩺 MedCheck")
st.subheader("Medical Claim Verification System")

st.markdown(
    """
MedCheck verifies medical advice or health-related claims using  
**evidence-based retrieval and reasoning**.

⚠️ This system provides **informational guidance only** and is **not a substitute
for professional medical advice**.
"""
)

# =============================
# Load Resources
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

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "text": metadata.iloc[idx]["text"],
            "distance": float(distances[0][rank])
        })
    return results

# =============================
# Verification Agent (Rule-based mock)
# =============================
def verify_medical_claim(claim, evidence):
    combined_text = " ".join([e["text"].lower() for e in evidence])

    if "long-term" in combined_text or "excessive" in combined_text:
        return {
            "verdict": "Unsafe",
            "explanation": (
                "The retrieved medical evidence indicates that long-term or excessive use "
                "may increase serious health risks."
            )
        }

    return {
        "verdict": "Partially Accurate",
        "explanation": (
            "The claim is not fully supported by the available medical evidence."
        )
    }

# =============================
# Personalization Agent (v2)
# =============================
def personalize_risk(user_profile, evidence):
    warnings = []
    evidence_text = " ".join([e["text"].lower() for e in evidence])

    for condition in user_profile.get("known_conditions", []):
        c = condition.lower()

        if "kidney" in c and "kidney" in evidence_text:
            warnings.append(
                "You have a kidney-related condition, and the evidence mentions kidney damage risk."
            )

        if "liver" in c and ("liver" in evidence_text or "hepatic" in evidence_text):
            warnings.append(
                "You have a liver-related condition, and the evidence mentions liver toxicity risk."
            )

        if "heart" in c and "cardiovascular" in evidence_text:
            warnings.append(
                "You have a heart-related condition, and the evidence mentions cardiovascular risk."
            )

    for allergy in user_profile.get("allergies", []):
        if allergy.lower() in evidence_text:
            warnings.append(
                f"You are allergic to {allergy}, which appears in the medical evidence."
            )

    if user_profile.get("pregnancy_status", False):
        if any(word in evidence_text for word in ["pregnancy", "pregnant", "fetal"]):
            warnings.append(
                "You are pregnant, and the evidence includes pregnancy-related risks."
            )

    return warnings

# =============================
# Confidence Scoring
# =============================
def compute_confidence(verdict, evidence_count):
    base_confidence = {
        "Accurate": 0.80,
        "Partially Accurate": 0.65,
        "Misleading": 0.70,
        "Unsafe": 0.85
    }

    confidence = base_confidence.get(verdict, 0.60)
    confidence += min(0.05, 0.01 * evidence_count)

    return round(min(confidence, 0.95), 2)

# =============================
# Optional User Profile (v2 UI)
# =============================
st.markdown("## 👤 Optional: Personal Health Profile")

with st.expander("Add personal medical details (optional)"):
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])

    conditions = st.text_input(
        "Known medical conditions (comma-separated)",
        placeholder="e.g. kidney disease, diabetes"
    )

    allergies = st.text_input(
        "Allergies (comma-separated)",
        placeholder="e.g. penicillin"
    )

    pregnancy_status = st.checkbox("Pregnant (if applicable)")

# Build user profile from UI
user_profile = None
if age > 0 or conditions.strip() or allergies.strip() or pregnancy_status:
    user_profile = {
        "age": age if age > 0 else None,
        "gender": gender.lower() if gender != "Prefer not to say" else None,
        "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
        "allergies": [a.strip().lower() for a in allergies.split(",") if a.strip()],
        "current_medications": [],
        "pregnancy_status": pregnancy_status
    }

# =============================
# Claim Input
# =============================
claim = st.text_area(
    "Enter a medical claim or advice",
    height=120,
    placeholder="Example: It is safe to take ibuprofen every day for a long time."
)

if st.button("Verify Claim"):
    if not claim.strip():
        st.warning("Please enter a medical claim.")
    else:
        with st.spinner("Verifying claim using medical evidence..."):
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

        if user_profile:
            warnings = personalize_risk(user_profile, evidence)
            if warnings:
                st.markdown("## ⚠️ Personalized Safety Warnings")
                for w in warnings:
                    st.warning(w)

        st.markdown("## 📚 Supporting Medical Evidence")
        for item in evidence:
            st.markdown(f"- {item['text']}")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption(
    "⚠️ MedCheck v2 provides informational guidance only and does not replace professional medical advice. "
    "Always consult a qualified healthcare provider."
)
