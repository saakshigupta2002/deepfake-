import streamlit as st
from transformers import AutoModel, AutoTokenizer, pipeline
import torch
from PIL import Image
import io
import traceback
import random
import base64
import os
import tempfile

# Set page config
st.set_page_config(
    page_title="Deepfake Analyzer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
        padding: 0 !important;
    }
    
    /* Header styling */
    .header-container {
        background-color: #1e2130;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat container styling */
    .chat-container {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* Message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #2e3a54;
    }
    .chat-message.bot {
        background-color: #3d4663;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        background-color: #4d5b82;
        color: white;
        font-size: 20px;
    }
    .chat-message .message {
        color: #fff;
        max-width: calc(100% - 52px);
    }
    
    /* Image styling */
    .chat-image {
        max-width: 300px;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border: 2px solid #3d4663;
    }
    
    /* Input area styling */
    .input-container {
        background-color: #1e2130;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #5e6bfd;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4c59e6;
        box-shadow: 0 0 10px rgba(94, 107, 253, 0.5);
    }
    .clear-button button {
        background-color: #fd5e5e !important;
    }
    .clear-button button:hover {
        background-color: #e64c4c !important;
        box-shadow: 0 0 10px rgba(253, 94, 94, 0.5) !important;
    }
    
    /* File uploader styling */
    .upload-area {
        border: 2px dashed #5e6bfd;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        background-color: rgba(94, 107, 253, 0.05);
    }
    .upload-area:hover {
        background-color: rgba(94, 107, 253, 0.1);
    }
    
    /* Custom file upload button */
    .custom-file-upload {
        display: inline-block;
        padding: 10px 15px;
        cursor: pointer;
        background-color: #5e6bfd;
        color: white;
        border-radius: 5px;
        margin-top: 10px;
        text-align: center;
    }
    .custom-file-upload:hover {
        background-color: #4c59e6;
    }
    
    /* Model selector styling */
    .stSelectbox > div > div {
        background-color: #2e3a54;
        color: white;
    }
    
    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] > div:first-child {
        background-color: #2e3a54;
        border-color: #5e6bfd;
    }
    
    /* File upload button - hide the default one */
    .stFileUploader > div > button {
        display: none;
    }
    .stFileUploader > div > small {
        display: none;
    }
    
    /* Image preview area */
    .image-preview {
        margin-top: 1rem;
        text-align: center;
    }
    .image-preview img {
        max-width: 100%;
        border-radius: 5px;
        border: 2px solid #3d4663;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("""
<div class="header-container">
    <h1 style="color: #fff; text-align: center; margin-bottom: 0.5rem;">Deepfake Analyzer</h1>
    <p style="text-align: center; color: #aaa; font-size: 1.2rem;">
        Chat with an AI assistant to analyze images for potential deepfakes
    </p>
</div>
""", unsafe_allow_html=True)

# Create two columns for main layout
col1, col2 = st.columns([3, 1])

###############################################################################
#                            LOADING THE MODEL                                #
###############################################################################
@st.cache_resource
def load_model(model_name):
    """Load a mock or real Hugging Face model and tokenizer."""
    # If using the deepfake explainer model, use a dummy pipeline
    if model_name == "saakshigupta/deepfake-explainer":
        
        class DummyDeepfakeExplainer:
            def __init__(self):
                self.indicators = [
                    "inconsistent lighting",
                    "unnatural skin texture",
                    "irregular facial features",
                    "misaligned elements",
                    "unusual blending boundaries",
                    "inconsistent shadows",
                    "artificial smoothing",
                    "unusual facial proportions",
                ]
                
            def generate(self, text, image=None):
                if image is not None:
                    img_hash = 0
                    if isinstance(image, str) and os.path.exists(image):
                        img_hash = sum(os.path.getsize(image) % 256 for _ in range(10))
                    # Randomly decide if it's fake or not
                    is_fake = (img_hash % 100) > 40
                    confidence = random.randint(65, 98)
                    selected_indicators = random.sample(self.indicators, random.randint(2, 3))
                    
                    if is_fake:
                        return (
                            f"Analysis result: This image appears to be a **POTENTIAL DEEPFAKE** "
                            f"with {confidence}% confidence level.\n\n"
                            f"Key indicators:\n"
                            + "".join([f"- {ind}\n" for ind in selected_indicators])
                            + "\n(Disclaimer: This is a mock analysis.)"
                        )
                    else:
                        return (
                            f"Analysis result: This image appears to be **AUTHENTIC** "
                            f"with {confidence}% confidence.\n\n"
                            "No major manipulation markers detected.\n\n"
                            "(Disclaimer: This is a mock analysis.)"
                        )
                else:
                    # Just a default text response
                    if any(x in text.lower() for x in ["what is", "how do", "explain"]):
                        return (
                            "Deepfakes are synthetic media where a person's likeness is replaced "
                            "using AI, especially GANs. Common indicators of deepfakes:\n"
                            "- Unnatural eye movements/blinks\n"
                            "- Inconsistent skin tone\n"
                            "- Irregular facial features\n"
                            "- Lip-sync issues\n"
                            "- Strange lighting or shadows\n\n"
                            "Feel free to upload an image to analyze."
                        )
                    else:
                        return (
                            "I'm a deepfake analysis assistant. Upload an image or ask me about "
                            "deepfake technology."
                        )
        
        # Dummy tokenizer that does nothing special
        class DummyTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [0, 1, 2], "attention_mask": [1, 1, 1]}
        
        return DummyDeepfakeExplainer(), DummyTokenizer()
    
    # Otherwise, load a standard model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-generation", model=model_name)
    return model, tokenizer

def generate_response(prompt, image_path=None, model_name="saakshigupta/deepfake-explainer"):
    """Generate a response using our dummy or real model."""
    model, tokenizer = load_model(model_name)
    
    # If it's our dummy deepfake model:
    if model_name == "saakshigupta/deepfake-explainer":
        if hasattr(model, "generate"):
            return model.generate(prompt, image_path)
        else:
            return "Sorry, I couldn’t generate a response using the deepfake explainer model."
    
    # If it's a standard text generation pipeline (GPT-2, etc.)
    # (In your original code you had a more complex logic, but let's keep it simple.)
    outputs = model(prompt, max_length=100, num_return_sequences=1)
    return outputs[0]["generated_text"]

###############################################################################
#                       SESSION STATE & UI LAYOUT                             #
###############################################################################
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Temporary directory to store uploaded images
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

with col2:
    st.markdown("<h3 style='color: #fff;'>Settings</h3>", unsafe_allow_html=True)
    model_option = st.selectbox(
        'Select Model',
        ["saakshigupta/deepfake-explainer", "gpt2", "facebook/bart-large-cnn"],
        index=0
    )
    
    st.markdown("<h3 style='color: #fff; margin-top: 1.5rem;'>Upload Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"], 
                                     label_visibility="collapsed",
                                     key="file_uploader")
    
    if uploaded_file is not None:
        try:
            temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            image = Image.open(temp_file_path)
            st.session_state.current_image_path = temp_file_path
            
            # Show preview
            st.markdown("<div class='image-preview'>", unsafe_allow_html=True)
            st.image(image, caption="Image Preview", use_column_width=True)
            st.success("✅ Image ready to send")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.session_state.current_image_path = None
    else:
        st.session_state.current_image_path = None
        st.markdown("""
        <div class="upload-area">
            <svg width="50" height="50" viewBox="0 0 24 24" fill="none"
                 xmlns="http://www.w3.org/2000/svg">
                <path d="M12 16.5V6.5M12 6.5L8 10.5M12 6.5L16 10.5"
                      stroke="#5e6bfd" stroke-width="2" stroke-linecap="round"
                      stroke-linejoin="round"/>
                <path d="M22 12C22 17.5228 17.5228 22 12 22
                         C6.47715 22 2 17.5228 2 12
                         C2 6.47715 6.47715 2 12 2
                         C17.5228 2 22 6.47715 22 12Z"
                      stroke="#5e6bfd" stroke-width="2"/>
            </svg>
            <p style="color: #fff; margin-top: 1rem;">Drag and drop an image here</p>
            <p style="color: #aaa; font-size: 0.8rem;">JPG, JPEG, PNG (max 5MB recommended)</p>
        </div>
        """, unsafe_allow_html=True)

###############################################################################
#                           DISPLAY EXISTING CHAT                              #
###############################################################################
with col1:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # If no messages in session, display welcome
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">
                <p>Hello! I'm a deepfake analysis assistant. I can help explain how deepfakes work and analyze images for signs of manipulation.</p>
                <p>How can I help you today?</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render the conversation
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            role_class = "user"
            avatar = "👤"
        else:
            role_class = "bot"
            avatar = "🤖"
        
        st.markdown(f"""
        <div class="chat-message {role_class}">
            <div class="avatar">{avatar}</div>
            <div class="message">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # If there's an image path attached
        if "image_path" in msg and msg["image_path"] is not None:
            try:
                img = Image.open(msg["image_path"])
                st.image(img, use_column_width=True)
            except Exception as e:
                st.error(f"Could not display image: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===================== Input area =====================
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    
    user_input = st.text_input(
        "",
        placeholder="Type your message here...",
        label_visibility="collapsed"
    )
    
    # Button row
    col_send, col_clear = st.columns([6, 1])
    
    with col_send:
        send_pressed = st.button("Send Message")
    with col_clear:
        st.markdown("<div class='clear-button'>", unsafe_allow_html=True)
        clear_pressed = st.button("Clear Chat")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== Handle user input =====================
if send_pressed:
    # If no text, but there's an image, create a default prompt
    if not user_input and st.session_state.current_image_path is not None:
        user_input = "Please analyze this image for possible deepfake manipulation."
    
    # If we still have no text & no image, do nothing
    if not user_input and st.session_state.current_image_path is None:
        st.warning("Please type a message or upload an image before pressing Send.")
    else:
        # Save user message in session
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "image_path": st.session_state.current_image_path
        })

        # Generate bot response
        with st.spinner("Analyzing..."):
            bot_answer = generate_response(
                prompt=user_input,
                image_path=st.session_state.current_image_path,
                model_name=model_option
            )

        # Append bot message
        st.session_state.messages.append({
            "role": "bot",
            "content": bot_answer,
            "image_path": None  # The bot doesn't send an image
        })
        
        # Clear the current image path after sending
        st.session_state.current_image_path = None
        
        # Force re-render
        st.rerun()

# ===================== Handle clear chat =====================
if clear_pressed:
    st.session_state.messages = []
    st.rerun()

# Cleanup temp files on app shutdown
def cleanup():
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        for file in os.listdir(st.session_state.temp_dir):
            os.remove(os.path.join(st.session_state.temp_dir, file))
        os.rmdir(st.session_state.temp_dir)

import atexit
atexit.register(cleanup)
