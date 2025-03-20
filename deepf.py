import streamlit as st
from PIL import Image
import torch
import io
import os
import traceback

# Set page config
st.set_page_config(page_title="Deepfake Analysis Bot", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E88E5;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #E3F2FD;
    }
    .chat-message.bot {
        background-color: #F5F5F5;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
    }
    .user-avatar, .bot-avatar {
        font-size: 1.5rem;
        min-width: 40px;
        margin-right: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .results-container {
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Deepfake Analysis Assistant")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "default_prompt" not in st.session_state:
    st.session_state.default_prompt = "Is this a deepfake? Analyze this image and provide both technical and non-technical explanations."

# Function to load model with detailed error reporting and mllama fix
@st.cache_resource(show_spinner=False)  # Turn off the built-in spinner since we use our own
def load_model(model_path, quantize, use_cpu):
    try:
        # Import necessary libraries
        from transformers import AutoProcessor, AutoModelForCausalLM
        import json
        import tempfile
        import os
        from huggingface_hub import hf_hub_download
        
        # Set up device mapping
        device_map = "auto" if not use_cpu and torch.cuda.is_available() else "cpu"
        st.info(f"Using device map: {device_map}")
        
        # First load the processor
        st.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Special handling for the 'mllama' issue - download config.json directly
        try:
            st.info("Loading model...")
            # Create a temporary directory for config
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try to download the config file
                config_path = hf_hub_download(
                    repo_id=model_path,
                    filename="config.json",
                    local_dir=temp_dir
                )
                
                # Read and modify the config
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Change model_type if it's 'mllama'
                if config_dict.get('model_type') == 'mllama':
                    st.info("Fixing model_type in config (changing 'mllama' to 'llama')")
                    config_dict['model_type'] = 'llama'
                    
                    # Save to a new temp file
                    modified_config_path = os.path.join(temp_dir, "modified_config.json")
                    with open(modified_config_path, 'w') as f:
                        json.dump(config_dict, f, indent=2)
                    
                    # Create model loading kwargs
                    model_loading_kwargs = {
                        "device_map": device_map,
                        "trust_remote_code": True,
                        "config": modified_config_path  # Use our modified config
                    }
                else:
                    # Standard loading kwargs
                    model_loading_kwargs = {
                        "device_map": device_map,
                        "trust_remote_code": True
                    }
                
                # Add precision setting
                if quantize and torch.cuda.is_available():
                    model_loading_kwargs["torch_dtype"] = torch.float16
                
                # Load the model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_loading_kwargs
                )
                
                return model, processor
                
        except Exception as config_err:
            # Fallback method: try with a forced configuration
            st.info("Using fallback method to load model...")
            from transformers import LlamaConfig, LlamaForCausalLM
            
            # Create a basic Llama config
            config = LlamaConfig.from_pretrained(model_path)
            # Ensure model_type is set to 'llama'
            config.model_type = 'llama'
            
            # Model loading kwargs
            model_loading_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True,
                "config": config
            }
            
            # Add precision setting
            if quantize and torch.cuda.is_available():
                model_loading_kwargs["torch_dtype"] = torch.float16
            
            # Try to load with our forced config
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                **model_loading_kwargs
            )
            
            return model, processor
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Full error details:")
        st.code(traceback.format_exc())
        return None, None

# Sidebar for model settings
with st.sidebar:
    st.header("Model Settings")
    
    # Use your fine-tuned model
    model_path = st.text_input("Model Path", "saakshigupta/deepfake-explainer-llama3-vision")
    
    col1, col2 = st.columns(2)
    with col1:
        quantize = st.checkbox("Use half precision", value=True)
    with col2:
        use_cpu = st.checkbox("Use CPU only", value=not torch.cuda.is_available())
    
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=1000, value=500, step=50)
    
    # Add a device info section
    st.subheader("Device Info")
    if torch.cuda.is_available():
        st.success(f"GPU available: {torch.cuda.get_device_name(0)}")
        if hasattr(torch.version, 'cuda'):
            st.info(f"CUDA Version: {torch.version.cuda}")
    else:
        st.warning("No GPU detected - will use CPU (slower)")
    
    # Force model reloading
    force_reload = st.checkbox("Force model reload", value=False)
    
    # Load model button
    if st.button("Load Model"):
        with st.spinner("Loading model... this might take a minute..."):
            # Clear cache if force reload is selected
            if force_reload and hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            
            # Load the model
            model, processor = load_model(model_path, quantize, use_cpu)
            
            if model is not None and processor is not None:
                st.session_state.model_loaded = True
                st.session_state.model = model
                st.session_state.processor = processor
                st.success("Model loaded successfully!")
            else:
                st.session_state.model_loaded = False
                st.error("Failed to load model. Please check the error details above.")
    
    # Reset chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.info("Chat history cleared!")

# Main content area
st.markdown("### Upload an image and ask about potential deepfakes")

# Display model status
if st.session_state.model_loaded:
    st.success("Model loaded and ready to analyze images")
else:
    st.warning("Please load the model first using the sidebar")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png"])

# Custom prompt with default value
custom_prompt = st.text_area("Your question (optional)", 
                            value=st.session_state.default_prompt, 
                            height=100,
                            help="You can customize your question or use the default prompt")

# Analysis button
analyze_button = st.button("Analyze Image", disabled=not (uploaded_file and st.session_state.model_loaded))

# Display chat history
st.markdown("### Conversation History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user" and "image" in message:
            st.image(message["image"], caption="Uploaded Image", use_column_width=True)
        st.write(message["content"])

# Handle image upload and prompt
if analyze_button:
    # Use the custom prompt or default
    prompt = custom_prompt if custom_prompt else st.session_state.default_prompt
    
    # Process the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Add user message with image to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image": image
    })
    
    # Display user message
    with st.chat_message("user"):
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image... This may take a moment..."):
            try:
                # Format the input for the model
                inputs = st.session_state.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                )
                
                # Move inputs to GPU if available
                if torch.cuda.is_available() and not use_cpu:
                    inputs = {k: v.to(st.session_state.model.device) for k, v in inputs.items()}
                
                # Generate response
                message_placeholder = st.empty()
                
                # Generate with appropriate parameters
                output_ids = st.session_state.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True
                )
                
                # Decode the output
                response = st.session_state.processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Clean up any model-specific tokens
                response = response.replace("<|eot_id|>", "").strip()
                
                # Display the response
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.error("Full error details:")
                st.code(traceback.format_exc())
                st.info("Troubleshooting tips: Try adjusting model settings or using a different image.")

# Additional information
with st.expander("About this app"):
    st.write("""
    ### Deepfake Analysis Assistant
    
    This application uses a fine-tuned Llama 3.2 Vision model to analyze images for potential deepfake manipulation.
    
    For each image, the model provides:
    - A technical explanation suitable for forensic experts
    - A non-technical explanation for general audiences
    
    #### How to use:
    1. Load the model using the sidebar
    2. Upload an image to analyze
    3. Ask a question about the image (or use the default prompt)
    4. View the analysis results
    
    #### Tips for best results:
    - Use clear, high-quality images
    - Images containing faces work best
    - Be specific in your questions
    """)

# Footer
st.markdown("---")
st.markdown("Deepfake Analysis Assistant | Created using Streamlit and Fine-tuned Llama 3.2 Vision")
