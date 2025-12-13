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
# UI Interaction
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

        st.markdown("## 📚 Supporting Medical Evidence")
        for item in evidence:
            st.markdown(f"- {item['text']}")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption(
    "MedCheck v1 • Agentic RAG-based Medical Verification System"
)
