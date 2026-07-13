import streamlit as py_streamlit
import pickle
import os

# 1. Configure page settings and application layouts
py_streamlit.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom CSS for UI styling
py_streamlit.markdown("""
    <style>
    .stApp {
        background-image: url("https://github.com/Aditya5929/Fake-News-Detector/blob/main/screenshots/background.png?raw=true");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
    }
    textarea {
        background-color: #252538 !important;
        color: #ffffff !important;
        border: 1px solid #4a4a6a !important;
        border-radius: 10px !important;
    }
    h1 {
        color: #00f2fe !important;
        text-shadow: 0px 0px 10px rgba(0, 242, 254, 0.5);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .subtitle-text {
        color: #b3b3cc;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Render Titles
py_streamlit.title(" Fake News Detection System")
py_streamlit.markdown('<p class="subtitle-text">Enter a news article below to check if it is Real or Fake in real-time.</p>', unsafe_allow_html=True)

# 🛠️ STATE MANAGEMENT FOR FIXED RESET
# We use a clearing counter key to completely force recreate the widget when button is clicked
if "clear_counter" not in py_streamlit.session_state:
    py_streamlit.session_state["clear_counter"] = 0

# Dynamic key generation based on counter
text_area_key = f"news_input_{py_streamlit.session_state['clear_counter']}"

# 2. Text Input Widget
user_input = py_streamlit.text_area("Paste the news text here:", height=150, key=text_area_key)

# Multi-column layout for horizontal buttons
col1, col2 = py_streamlit.columns([1, 4])

with col1:
    submit_button = py_streamlit.button("Check News", type="primary")

with col2:
    # Clicking this increments counter, which forces dynamic key to shift and flushes input text instantly
    clear_button = py_streamlit.button("Clear Text")
    if clear_button:
        py_streamlit.session_state["clear_counter"] += 1
        py_streamlit.rerun()

# 3. Model Inference Phase
if submit_button:
    if user_input.strip() == "":
        py_streamlit.warning("Please enter some text first!")
    else:
        # Validate existence of configuration files
        if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
            py_streamlit.error("❌ Error: Missing weights config! Please run 'python train_model.py' in your terminal first.")
        else:
            # Load trained weights
            with open('model.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            with open('vectorizer.pkl', 'rb') as vec_file:
                vectorizer = pickle.load(vec_file)
            
            # Execute predictions pipeline
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)[0]
            
            # Render outputs
            py_streamlit.markdown("---")
            if prediction == 1:
                py_streamlit.error("🚨 WARNING: This news is likely FAKE!")
            else:
                py_streamlit.success("✅ RELIABLE: This news appears to be REAL.")