import streamlit as st
import base64
from PIL import Image
import requests
import importlib.util
import sys
import os

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:5000/authenticate"

# -----------------------------
# Session State
# -----------------------------
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "auth_passed" not in st.session_state:
    st.session_state.auth_passed = False

# -----------------------------
# Agents
# -----------------------------
agents = [
    {"name": "Claim Validator", "file": "Claim_Validator.py"},
    {"name": "Mood Detection", "file": "detect_mood.py"},
    {"name": "Quiz Maker", "file": "quiz_maker.py"}
]

# -----------------------------
# Login
# -----------------------------
st.title("Face Authentication Login")
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file and not st.session_state.auth_passed:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, width=250)

    if st.button("Authenticate"):
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        try:
            response = requests.post(API_URL, json={"image": img_b64})
            data = response.json()
            user_name = data.get("prediction", "User")
            st.session_state.user_name = user_name

            if data.get("status") == "success" and data.get("authorized") == "approved":
                st.success(f"✅ Access approved! Welcome, {user_name} 🎉")
                st.session_state.auth_passed = True
            else:
                st.error(f"❌ Access denied, {user_name}")
        except Exception as e:
            st.error(f"API request failed: {e}")

# -----------------------------
# Agent Selection
# -----------------------------
if st.session_state.auth_passed and st.session_state.current_agent is None:
    st.subheader("Select an Agent:")
    for idx, agent in enumerate(agents):
        if st.button(f"Access {agent['name']}", key=f"btn{idx}"):
            st.session_state.current_agent = agent

# -----------------------------
# Run Selected Agent
# -----------------------------
if st.session_state.current_agent:
    agent = st.session_state.current_agent
    st.header(f"🧩 Agent: {agent['name']}")

    if st.button("Run Agent"):
        try:
            # Dynamically import module
            spec = importlib.util.spec_from_file_location("agent_module", agent["file"])
            module = importlib.util.module_from_spec(spec)
            sys.modules["agent_module"] = module
            spec.loader.exec_module(module)
            # Call run() function
            module.run()
        except Exception as e:
            st.error(f"Failed to run agent: {e}")

    if st.button("⬅ Back to Agents"):
        st.session_state.current_agent = None
