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
    if not email or not password:
        st.error("Email and password are required.")
    else:
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
                st.success("Registration successful! Please login.")
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
                    st.success("Logged in successfully!")
                else:
                    st.error("User profile not found.")
            else:
                st.error(res.get("error", {}).get("message", "Login failed"))

if not st.session_state.logged_in:
    st.info("Please login or register to use MedCheck.")
    st.stop()

# =========================================================
# EDIT PROFILE (RESTORED)
# =========================================================
st.sidebar.markdown("## 👤 Profile Settings")
profile = st.session_state.user_profile

with st.sidebar.form("edit_profile"):
    age = st.number_input("Age", 0, 120, value=profile.get("age", 0))
    gender = st.selectbox(
        "Gender",
        ["Prefer not to say", "Male", "Female", "Other"],
        index=["Prefer not to say", "Male", "Female", "Other"].index(
            profile.get("gender", "Prefer not to say")
        )
    )
    conditions = st.text_input(
        "Known medical conditions",
        ", ".join(profile.get("known_conditions", []))
    )
    allergies = st.text_input(
        "Allergies",
        ", ".join(profile.get("allergies", []))
    )
    pregnancy = st.checkbox(
        "Pregnant (if applicable)",
        value=profile.get("pregnancy_status", False)
    )

    save = st.form_submit_button("💾 Save Profile")

if save:
    updated = {
        "email": profile.get("email"),
        "age": age,
        "gender": gender,
        "known_conditions": [c.strip().lower() for c in conditions.split(",") if c.strip()],
        "allergies": [a.strip().lower() for a in allergies.split(",") if a.strip()],
        "pregnancy_status": pregnancy
    }
    db.collection("users").document(st.session_state.user_uid).set(updated)
    st.session_state.user_profile = updated
    st.sidebar.success("Profile updated successfully!")

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
    _, indices = index.search(emb, top_k)
    return [{"text": metadata.iloc[idx]["text"]} for idx in indices[0]]

# =========================================================
# Entity-Relevant Evidence Filter (FIXED)
# =========================================================
def filter_evidence_by_query_entities(query, evidence):
    query = query.lower()
    key_entities = [
        "paracetamol", "acetaminophen",
        "ibuprofen", "aspirin",
        "antibiotic", "antibiotics",
        "benzodiazepine", "opioid",
        "milk", "lemon"
    ]
    matched = [e for e in key_entities if e in query]

    if not matched:
        return evidence[:2]

    filtered = [e for e in evidence if any(m in e["text"].lower() for m in matched)]
    return filtered[:3] if filtered else evidence[:2]

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
# 🔒 FROZEN v4.0 LOGIC
# =========================================================
def classify_duration(claim):
    c = claim.lower()
    if any(x in c for x in ["2 days", "3 days", "few days", "once", "short term"]):
        return "short"
    if any(x in c for x in ["months", "weeks", "every day", "daily", "long term"]):
        return "long"
    return "unknown"

def extract_evidence_signals(evidence):
    text = " ".join(e["text"].lower() for e in evidence)
    return {
        "warns_normal": any(p in text for p in [
            "should not be used", "contraindicated",
            "not safe", "do not use", "do not treat"
        ]),
        "warns_excessive": any(p in text for p in [
            "long-term", "long term", "prolonged",
            "excessive", "toxicity", "risk of"
        ])
    }

def decide_verdict_v4(duration, analysis):
    if analysis["warns_normal"]:
        return "Unsafe"
    if duration == "long" and analysis["warns_excessive"]:
        return "Unsafe"
    return "Safe"

def generate_explanation(claim, evidence, verdict):
    ev = "\n".join(f"- {e['text']}" for e in evidence)
    prompt = f"""
Use ONLY the medical evidence below.
Explain why the verdict is {verdict}.

Claim:
{claim}

Evidence:
{ev}

Explanation (2–3 sentences):
"""
    inp = llm_tokenizer(prompt, return_tensors="pt", truncation=True)
    out = llm_model.generate(**inp, max_new_tokens=100)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True)

def verify_medical_claim_llm(claim, evidence):
    duration = classify_duration(claim)
    analysis = extract_evidence_signals(evidence)
    verdict = decide_verdict_v4(duration, analysis)
    explanation = generate_explanation(claim, evidence, verdict)
    return verdict, explanation

# =========================================================
# UI – Claim Verification
# =========================================================
st.markdown("## 🧠 Verify Medical Claim")
claim = st.text_area("Enter a medical claim", height=120)

if st.button("Verify Claim"):
    raw_evidence = retrieve_medical_evidence(claim)
    evidence = filter_evidence_by_query_entities(claim, raw_evidence)

    verdict, explanation = verify_medical_claim_llm(claim, evidence)

    st.markdown("## 🧾 Verdict")
    if verdict == "Safe":
        st.success(verdict)
    else:
        st.error(verdict)

    st.markdown("## 🔍 Explanation")
    st.write(explanation)

    # =====================================================
    # Personalized Safety Assessment
    # =====================================================
    st.markdown("## ⚠️ Personalized Safety Assessment")
    evidence_text = " ".join(e["text"].lower() for e in evidence)
    warnings = []

    for condition in st.session_state.user_profile.get("known_conditions", []):
        if "kidney" in condition and "kidney" in evidence_text:
            warnings.append("Your kidney condition may increase risk.")
        if "liver" in condition and "liver" in evidence_text:
            warnings.append("Your liver condition may increase risk.")
        if "heart" in condition and "cardiovascular" in evidence_text:
            warnings.append("Your heart condition may increase risk.")

    if st.session_state.user_profile.get("pregnancy_status", False):
        warnings.append("Pregnancy may increase medication-related risks.")

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
