
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import torch
from threading import Thread
from pathlib import Path
import time
import json
import random
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.let_it_rain import rain
from streamlit_extras.function_explorer import function_explorer
from streamlit_extras.add_vertical_space import add_vertical_space
import numpy as np
import pyttsx3 
import base64  
from io import BytesIO  

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 150)  
engine.setProperty('volume', 2.0)  
if len(voices) > 2:
    engine.setProperty('voice', voices[2].id) 

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = Path(r"E:\Pro\NLPCbot\tinyllama-finetuned-20250514_184430\final_model")

st.set_page_config(
    page_title="Oggy.AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Oggy - Enterprise AI Assistant with Voice"
    }
)
st.markdown("""
<style>
    :root {
        --primary: #8a2be2;
        --secondary: #ff6b6b;
        --accent: #00c6fb;
        --dark: #0f172a;
        --darker: #020617;
        --darkest: #01050e;
        --light: #f8fafc;
        --gray: #94a3b8;
        --dark-gray: #334155;
    }
    
    /* Voice control buttons */
    .voice-controls {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        align-items: center;
    }
    
    .voice-btn {
        background: rgba(255,255,255,0.1);
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        color: var(--light);
    }
    
    .voice-btn:hover {
        background: rgba(255,255,255,0.2);
        transform: scale(1.1);
    }
    
    .voice-btn.active {
        background: var(--primary);
        box-shadow: 0 0 10px rgba(138, 43, 226, 0.5);
    }
    
    .voice-btn i {
        font-size: 16px;
    }
    
    .voice-speed-control {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-left: auto;
    }
    
    .voice-speed-label {
        font-size: 12px;
        color: var(--gray);
    }
    
    /* Main app background - Dark theme */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, var(--darkest), var(--darker)) !important;
        color: var(--light) !important;
    }
    
    /* Chat container - Dark glass morphism */
    [data-testid="stChatMessageContainer"] {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(195deg, var(--darkest) 0%, var(--dark) 100%) !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.3) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    /* Chat input box - Dark theme */
    .stChatInput {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    .stChatInput textarea {
        min-height: 120px !important;
        border-radius: 16px !important;
        border: 1px solid var(--dark-gray) !important;
        padding: 16px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        background: rgba(2, 6, 23, 0.7) !important;
        color: var(--light) !important;
        backdrop-filter: blur(5px) !important;
    }
    
    .stChatInput textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(138, 43, 226, 0.3) !important;
        background: rgba(2, 6, 23, 0.9) !important;
    }
    
    /* Send button */
    .stChatInput button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 28px !important;
        margin-top: 14px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }
    
    .stChatInput button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(138, 43, 226, 0.4) !important;
    }
    
    /* User message bubble */
    [data-testid="stChatMessage-user"] {
        justify-content: flex-end !important;
    }
    
    [data-testid="stChatMessage-user"] div:first-child {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border-radius: 24px 24px 8px 24px !important;
        padding: 18px 22px !important;
        margin: 12px 0 !important;
        box-shadow: 0 4px 20px rgba(138, 43, 226, 0.3) !important;
        max-width: 75%;
        border: none !important;
        animation: messageIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    /* Assistant message bubble - Dark theme */
    [data-testid="stChatMessage-assistant"] {
        justify-content: flex-start !important;
    }
    
    [data-testid="stChatMessage-assistant"] div:first-child {
        background: rgba(2, 6, 23, 0.8) !important;
        color: var(--light) !important;
        border-radius: 24px 24px 24px 8px !important;
        padding: 18px 22px !important;
        margin: 12px 0 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
        max-width: 75%;
        border: none !important;
        border-left: 4px solid var(--primary) !important;
        animation: messageIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        backdrop-filter: blur(5px) !important;
    }
    
    /* Typing animation */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        background: rgba(2, 6, 23, 0.8);
        border-radius: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-left: 8px;
        backdrop-filter: blur(5px);
    }
    
    .typing-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: var(--primary);
        margin-right: 6px;
        animation: typingAnimation 1.6s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typingAnimation {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
        30% { transform: translateY(-6px); opacity: 1; }
    }
    
    @keyframes messageIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Header styling - Dark theme */
    .header-container {
        background: linear-gradient(135deg, var(--dark), var(--darkest));
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .header-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(138, 43, 226, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        position: relative;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
        font-weight: 400;
        position: relative;
    }
    
    /* Sidebar elements - Dark theme */
    .sidebar-title {
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        position: relative;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .sidebar-subtitle {
        color: rgba(255,255,255,0.7) !important;
        font-size: 1rem !important;
        margin-bottom: 2rem !important;
        position: relative;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        background: rgba(255,255,255,0.08);
    }
    
    .metric-title {
        color: var(--gray);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    
    /* Floating action button */
    .floating-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 100;
    }
    
    .floating-btn:hover {
        transform: translateY(-3px) scale(1.1);
        box-shadow: 0 8px 24px rgba(138, 43, 226, 0.4);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary);
    }
    
    /* Message timestamp */
    .message-timestamp {
        font-size: 0.7rem;
        color: var(--gray);
        margin-top: 4px;
        text-align: right;
    }
    
    /* Glow effect for important messages */
    .glow {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 0 5px rgba(138, 43, 226, 0.5);
        }
        to {
            box-shadow: 0 0 20px rgba(138, 43, 226, 0.8);
        }
    }
    
    /* Loyalty badge styling */
    .loyalty-badge {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(138, 43, 226, 0.2);
    }
    
    .loyalty-badge::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    .loyalty-title {
        color: white;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        position: relative;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .loyalty-name {
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        position: relative;
        display: flex;
        align-items: center;
    }
    
    .loyalty-initials {
        background: rgba(0,0,0,0.2);
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-weight: 800;
        font-size: 1.1rem;
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .loyalty-status {
        color: var(--accent);
        font-size: 0.8rem;
        font-weight: 600;
        position: relative;
        display: flex;
        align-items: center;
    }
    
    .loyalty-status::before {
        content: "";
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    
    /* VIP crown icon */
    .crown-icon {
        margin-left: 8px;
        font-size: 1.2rem;
        color: gold;
        filter: drop-shadow(0 0 2px rgba(255,215,0,0.5));
    }
    
    /* Audio player styling */
    .audio-player {
        width: 100%;
        margin-top: 8px;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* New 3D button effect */
    .btn-3d {
        position: relative;
        display: inline-block;
        padding: 12px 24px;
        border: none;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        cursor: pointer;
        box-shadow: 0 5px 0 rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }
    
    .btn-3d:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 0 rgba(0,0,0,0.2);
    }
    
    .btn-3d:active {
        transform: translateY(3px);
        box-shadow: 0 2px 0 rgba(0,0,0,0.2);
    }
    
    /* Animated background particles */
    .particle {
        position: absolute;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
    }
</style>
""", unsafe_allow_html=True)

def text_to_speech(text, speed=1.0):
    """Convert text to speech and return audio data"""
    engine.setProperty('rate', 150 * speed)  # Adjust speed
    
    # Save speech to a temporary file
    audio_file = "temp_audio.wav"
    engine.save_to_file(text, audio_file)
    engine.runAndWait()
    
    # Read the audio file and convert to base64
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    
    return audio_bytes

def autoplay_audio(audio_bytes):
    """Create an HTML audio player with autoplay"""
    audio_str = base64.b64encode(audio_bytes).decode('utf-8')
    audio_html = f"""
        <audio autoplay controls class="audio-player">
            <source src="data:audio/wav;base64,{audio_str}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def typing_animation():
    return """
    <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
    </div>
    """

def render_header():
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">Oggy</h1>
        <p class="header-subtitle">Enterprise-grade AI Assistant with Voice Interaction</p>
    </div>
    """, unsafe_allow_html=True)

def confetti_effect():
    rain(
        emoji="✨",
        font_size=30,
        falling_speed=5,
        animation_length=1
    )

def get_timestamp():
    return time.strftime("%H:%M", time.localtime())

# model load krna hai
@st.cache_resource(show_spinner=False)
def load_peft_model():
    with st.spinner("🚀 Initializing AI engine..."):
        try:
            
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            
            model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            
            return model, tokenizer
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            st.stop()

model, tokenizer = load_peft_model()

def generate_response(prompt, temperature=0.7, max_length=256):
    
    messages = [{"role": "user", "content": prompt}]
    
    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)
    
    
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(
        inputs=inputs,
        streamer=streamer,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    
    
    full_response = ""
    message_placeholder = st.empty()
    
   
    message_placeholder.markdown(
        f'<div style="display:flex;align-items:center;">{typing_animation()}</div>', 
        unsafe_allow_html=True
    )
    
    
    for chunk in streamer:
        full_response += chunk
        show_typing = not chunk.endswith((" ", "\n", ".", ",", "!", "?"))
        typing_html = typing_animation() if show_typing else ""
    
        typing_div = f'<div style="display:flex;align-items:center;">{typing_html}</div>' if show_typing else ""

        message_placeholder.markdown(
            f'''
            <div style="display:flex;flex-direction:column;gap:8px;">
               <div class="assistant-message">{full_response}</div>
               {typing_div}
               <div class="message-timestamp">{get_timestamp()}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    
    
    message_placeholder.markdown(
        f'<div style="display:flex;flex-direction:column;gap:8px;">'
        f'<div class="assistant-message">{full_response}</div>'
        f'<div class="message-timestamp">{get_timestamp()}</div>'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    return full_response
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = True
if 'voice_speed' not in st.session_state:
    st.session_state.voice_speed = 1.0

with st.sidebar:
    st.markdown('<p class="sidebar-title">Oggy.AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">Your enterprise AI assistant</p>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### ⚙️ Conversation Settings")
    temperature = st.slider(
        "Creativity Level", 
        0.1, 1.0, 0.7,
        help="Higher values produce more creative but less predictable responses"
    )
    max_length = st.slider(
        "Response Length", 
        64, 512, 256,
        help="Maximum number of tokens in generated responses"
    )
    
    st.divider()
    st.markdown("### 🔊 Voice Settings")
    st.session_state.voice_enabled = st.checkbox(
        "Enable Voice Output", 
        value=True,
        help="Enable text-to-speech for assistant responses"
    )
    st.session_state.voice_speed = st.slider(
        "Voice Speed", 
        0.5, 2.0, 1.0, 0.1,
        help="Adjust the speech rate of the voice output"
    )
    
    st.divider()
    st.markdown("### System Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">DEVICE</p>
            <p class="metric-value">{}</p>
        </div>
        """.format(model.device.type.upper()), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">PRECISION</p>
            <p class="metric-value">{}</p>
        </div>
        """.format("16-bit" if torch.cuda.is_available() else "32-bit"), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <p class="metric-title">MODEL</p>
        <p class="metric-value">TinyLlama 1.1B</p>
        <p style="margin:0;font-size:0.8rem;color:rgba(255,255,255,0.7);">Fine-tuned on OASST1</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Lakshya Pendharkar
    st.divider()
    st.markdown("""
    <div class="loyalty-badge">
        <div class="loyalty-title">Created By</div>
        <div class="loyalty-name">
            <span class="loyalty-initials">L.P.</span>
            Lakshya Pendharkar
            <span class="crown-icon"></span>
        </div>
        <div class="loyalty-status"> • Developer</div>
    </div>
    """, unsafe_allow_html=True)

render_header()

if "messages" not in st.session_state:
    welcome_messages = [
        "Hello Lakshya! I'm your Oggy. How can I help you today?",
        "Welcome back Lakshya! What would you like to discuss today?",
        "Greetings Lakshya! I'm ready to assist you."
    ]
    welcome_msg = random.choice(welcome_messages)
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_msg, "timestamp": get_timestamp()}
    ]
    if st.session_state.voice_enabled:
        audio_bytes = text_to_speech(welcome_msg, st.session_state.voice_speed)
        autoplay_audio(audio_bytes)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.markdown(f'<div class="message-timestamp">{message["timestamp"]}</div>', unsafe_allow_html=True)
        
        if message["role"] == "assistant":
            with st.container():
                st.markdown("""
                <div class="voice-controls">
                    <button class="voice-btn" onclick="playAudio(this)" title="Play">
                        <i>▶️</i>
                    </button>
                    <button class="voice-btn" onclick="pauseAudio(this)" title="Pause">
                        <i>⏸️</i>
                    </button>
                    <div class="voice-speed-label">Speed: {:.1f}x</div>
                </div>
                """.format(st.session_state.voice_speed), unsafe_allow_html=True)


st.markdown("""
<div class="floating-btn" onclick="alert('New conversation started!')">
    <span style="font-size: 1.5rem;">+</span>
</div>
""", unsafe_allow_html=True)

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": get_timestamp()})
    
    
    with st.chat_message("user"):
        st.markdown(prompt)
        st.markdown(f'<div class="message-timestamp">{get_timestamp()}</div>', unsafe_allow_html=True)
    
    
    with st.chat_message("assistant"):
        response = generate_response(
            prompt,
            temperature=temperature,
            max_length=max_length
        )
        
        if st.session_state.voice_enabled:
            audio_bytes = text_to_speech(response, st.session_state.voice_speed)
            autoplay_audio(audio_bytes)
            st.markdown("""
            <div class="voice-controls">
                <button class="voice-btn" onclick="playAudio(this)" title="Play">
                    <i>▶️</i>
                </button>
                <button class="voice-btn" onclick="pauseAudio(this)" title="Pause">
                    <i>⏸️</i>
                </button>
                <div class="voice-speed-label">Speed: {:.1f}x</div>
            </div>
            """.format(st.session_state.voice_speed), unsafe_allow_html=True)
    
    
    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": get_timestamp()})
    
    
    if len(st.session_state.messages) == 3:
        confetti_effect()
st.components.v1.html("""
<script>
// Enhanced Floating Button with Ripple Effect
const floatingBtn = document.querySelector('.floating-btn');
floatingBtn.addEventListener('click', function(e) {
    // Create ripple element
    const ripple = document.createElement('span');
    ripple.classList.add('ripple-effect');
    
    // Position ripple
    const rect = this.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    ripple.style.width = ripple.style.height = `${size}px`;
    ripple.style.left = `${e.clientX - rect.left - size/2}px`;
    ripple.style.top = `${e.clientY - rect.top - size/2}px`;
    
    // Add ripple to button
    this.appendChild(ripple);
    
    // Remove ripple after animation
    setTimeout(() => ripple.remove(), 600);
    
    // Create confetti explosion
    createConfetti(e.clientX, e.clientY);
    
    // Show elegant toast notification instead of alert
    showToast('New conversation started!', 'success');
});

// Audio control functions
function playAudio(btn) {
    const audio = btn.closest('.voice-controls').previousElementSibling.querySelector('audio');
    if (audio) {
        audio.play();
        btn.classList.add('active');
        setTimeout(() => btn.classList.remove('active'), 200);
    }
}

function pauseAudio(btn) {
    const audio = btn.closest('.voice-controls').previousElementSibling.querySelector('audio');
    if (audio) {
        audio.pause();
        btn.classList.add('active');
        setTimeout(() => btn.classList.remove('active'), 200);
    }
}

// Create confetti explosion effect
function createConfetti(x, y) {
    const colors = ['#8a2be2', '#ff6b6b', '#00c6fb', '#ffffff'];
    const confettiCount = 50;
    
    for (let i = 0; i < confettiCount; i++) {
        const confetti = document.createElement('div');
        confetti.classList.add('confetti');
        
        // Random properties
        const size = Math.random() * 10 + 5;
        const color = colors[Math.floor(Math.random() * colors.length)];
        const angle = Math.random() * Math.PI * 2;
        const velocity = Math.random() * 5 + 5;
        const spin = Math.random() * 10 - 5;
        
        // Set styles
        confetti.style.width = `${size}px`;
        confetti.style.height = `${size}px`;
        confetti.style.backgroundColor = color;
        confetti.style.left = `${x}px`;
        confetti.style.top = `${y}px`;
        confetti.style.transform = `rotate(${Math.random() * 360}deg)`;
        
        // Add to body
        document.body.appendChild(confetti);
        
        // Animate
        const animation = confetti.animate([
            { 
                transform: `translate(0, 0) rotate(0deg)`,
                opacity: 1 
            },
            { 
                transform: `translate(${Math.cos(angle) * velocity * 50}px, ${Math.sin(angle) * velocity * 50}px) rotate(${spin * 360}deg)`,
                opacity: 0 
            }
        ], {
            duration: 1000,
            easing: 'cubic-bezier(0.4, 0, 0.2, 1)'
        });
        
        // Remove after animation
        animation.onfinish = () => confetti.remove();
    }
}

// Elegant toast notification
function showToast(message, type) {
    const toast = document.createElement('div');
    toast.classList.add('toast-notification', type);
    toast.innerHTML = `
        <div class="toast-icon">${type === 'success' ? '✓' : '✕'}</div>
        <div class="toast-message">${message}</div>
    `;
    
    document.body.appendChild(toast);
    
    // Slide in
    toast.animate([
        { transform: 'translateY(20px)', opacity: 0 },
        { transform: 'translateY(0)', opacity: 1 }
    ], { duration: 300, easing: 'ease-out' });
    
    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        toast.animate([
            { transform: 'translateY(0)', opacity: 1 },
            { transform: 'translateY(-20px)', opacity: 0 }
        ], { 
            duration: 300,
            easing: 'ease-in'
        }).onfinish = () => toast.remove();
    }, 3000);
}

// Enhanced smooth scrolling with momentum
window.addEventListener('load', function() {
    const chatContainer = document.querySelector('[data-testid="stChatMessageContainer"]');
    if (chatContainer) {
        let isScrolling = false;
        
        // Initial scroll to bottom
        smoothScrollTo(chatContainer, chatContainer.scrollHeight, 800);
        
        // Watch for new messages
        const observer = new MutationObserver(() => {
            if (!isScrolling) {
                smoothScrollTo(chatContainer, chatContainer.scrollHeight, 500);
            }
        });
        
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
});

function smoothScrollTo(element, to, duration) {
    const start = element.scrollTop;
    const change = to - start;
    const startTime = performance.now();
    let isScrolling = true;
    
    function animateScroll(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeProgress = easeOutQuart(progress);
        element.scrollTop = start + change * easeProgress;
        
        if (progress < 1) {
            requestAnimationFrame(animateScroll);
        } else {
            isScrolling = false;
        }
    }
    
    requestAnimationFrame(animateScroll);
}

function easeOutQuart(t) {
    return 1 - Math.pow(1 - t, 4);
}

// Enhanced message glow effect with particle animation
setTimeout(function() {
    const messages = document.querySelectorAll('[data-testid="stChatMessage-assistant"] div:first-child');
    if (messages.length > 0) {
        messages[0].classList.add('glow');
        
        // Add floating particles to first message
        createMessageParticles(messages[0]);
    }
}, 1000);

function createMessageParticles(element) {
    const rect = element.getBoundingClientRect();
    const particleCount = 15;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.classList.add('message-particle');
        
        // Position randomly within message
        const x = Math.random() * rect.width;
        const y = Math.random() * rect.height;
        
        particle.style.left = `${x}px`;
        particle.style.top = `${y}px`;
        
        // Random size and delay
        const size = Math.random() * 4 + 2;
        const delay = Math.random() * 2;
        
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.animationDelay = `${delay}s`;
        
        element.appendChild(particle);
    }
}

// Add CSS for new effects
const style = document.createElement('style');
style.textContent = `
    /* Ripple effect */
    .ripple-effect {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.6);
        transform: scale(0);
        animation: ripple 600ms linear;
        pointer-events: none;
    }
    
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    /* Confetti particles */
    .confetti {
        position: fixed;
        pointer-events: none;
        z-index: 9999;
        border-radius: 50%;
    }
    
    /* Toast notification */
    .toast-notification {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(10px);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        z-index: 1000;
        transform: translateY(20px);
        opacity: 0;
    }
    
    .toast-notification.success {
        border-left: 4px solid #4ade80;
    }
    
    .toast-notification.error {
        border-left: 4px solid #f87171;
    }
    
    .toast-icon {
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Message particles */
    .message-particle {
        position: absolute;
        background: var(--primary);
        border-radius: 50%;
        pointer-events: none;
        animation: float 4s infinite ease-in-out;
        opacity: 0.6;
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0) translateX(0);
            opacity: 0.6;
        }
        50% {
            transform: translateY(-20px) translateX(10px);
            opacity: 0.9;
        }
    }
    
    /* Background particles */
    .particle {
        position: absolute;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
        animation: float 15s infinite linear;
    }
    
    /* Audio player custom styling */
    audio::-webkit-media-controls-panel {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    audio::-webkit-media-controls-play-button,
    audio::-webkit-media-controls-mute-button {
        filter: invert(1);
    }
    
    audio::-webkit-media-controls-current-time-display,
    audio::-webkit-media-controls-time-remaining-display {
        color: white;
        font-family: sans-serif;
    }
`;
document.head.appendChild(style);

// Create animated background particles
function createBackgroundParticles() {
    const particleCount = 30;
    const container = document.querySelector('[data-testid="stAppViewContainer"]');
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        
        // Random properties
        const size = Math.random() * 10 + 5;
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const opacity = Math.random() * 0.3 + 0.1;
        const duration = Math.random() * 20 + 10;
        const delay = Math.random() * 10;
        
        // Set styles
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${x}vw`;
        particle.style.top = `${y}vh`;
        particle.style.opacity = opacity;
        particle.style.animationDuration = `${duration}s`;
        particle.style.animationDelay = `-${delay}s`;
        
        // Add to container
        container.appendChild(particle);
    }
}

// Initialize background particles
createBackgroundParticles();
</script>
""", height=0)