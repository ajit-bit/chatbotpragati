import streamlit as st
import json, time, uuid
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import base64

load_dotenv()

# Load logo
def get_base64(path):
    try:
        return base64.b64encode(open(path, "rb").read()).decode()
    except FileNotFoundError:
        return None

logo64 = get_base64("TSPlogo.jpg")

# Chatbot detection
try:
    from chatbot_inference import ChatbotInference
    CHATBOT_AVAILABLE = True
except:
    CHATBOT_AVAILABLE = False

st.set_page_config(page_title="Pragati.TSP", layout="wide")

# Font Awesome CSS for markdown icons
st.markdown("""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# --- Styles ---
st.markdown("""
<style>
  /* Header */
  .main-header {
    background: linear-gradient(135deg, #0A1A2A, #13334C);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    color: white;
  }
  .main-header img {
    height: 50px;
    margin-right: 20px;
  }
  .main-header h1 {
    margin: 0;
    font-size: 1.6rem;
  }
  .main-header p {
    margin: 0;
    color: #ccc;
    font-size: 0.9rem;
  }

  /* Sidebar items */
  .sidebar-chat {
    padding: 0.6rem;
    border-radius: 6px;
    margin: 4px 0;
    text-align: left;
    color: #333;
    transition: all 0.25s ease;
  }
  .sidebar-chat:hover {
    background: #1A2E42;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #0A1A2A, #13334C);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 6px 12px;
    text-align: left;
    transition: all 0.25s ease;
  }
  .stButton > button:hover {
    background-color: #1A2E42;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
  }

  /* Chat Bubbles */
  .chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border: 1px solid #e0e0e0;
  }
  .user-message {
    background: linear-gradient(135deg, #0A1A2A, #13334C);
    color: white;
    margin-left: 2rem;
  }
  .bot-message {
    background: #f9f9f9;
    color: #333;
    margin-right: 2rem;
  }
</style>

""", unsafe_allow_html=True)

# --- Header with Logo ---
if logo64:
    st.markdown(f"""
    <div class="main-header">
      <img src="data:image/jpeg;base64,{logo64}">
      <div>
        <h1>Pragati.TSP</h1>
        <p>Market Research Assistant v1.0</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Logo not found. Please add TSPlogo.jpg to the directory.")

# Initialize chatbot
@st.cache_resource
def init_bot():
    if CHATBOT_AVAILABLE:
        try: return ChatbotInference()
        except: return None
    return None

bot = init_bot()

from health_check import display_health_status
display_health_status()

def fallback(inp):
    inp = inp.lower()
    if "market" in inp:
        return "I can help with market insights, trends, sizing..."
    if "competitor" in inp:
        return "Let’s analyze your competitors—features, pricing..."
    if "survey" in inp or "data" in inp:
        return "Sure! I can help with surveys and data."
    return "Let me know whether you'd like insights on market, competitor, or survey."

def chat_name(msg):
    m = msg.lower()
    if "market" in m: return "Market Research Chat"
    if "competitor" in m: return "Competitor Analysis Chat"
    if "survey" in m: return "Survey Chat"
    return "Pragati.TSP Chat"

if 'messages' not in st.session_state: st.session_state.messages = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state: st.session_state.current_chat_id = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("### <i class='fas fa-compass'></i> Explore", unsafe_allow_html=True)
    search = st.text_input("Search Chats...", placeholder="keyword...")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

    st.markdown("---")
    st.markdown("### <i class='fas fa-history'></i> Recent Chats", unsafe_allow_html=True)

    for cid, info in st.session_state.chat_history.items():
        if not search or search.lower() in info['name'].lower():
            c1, c2 = st.columns([6, 1])

            if c1.button(info['name'], key=f"r_{cid}"):
                st.session_state.messages = info['messages']
                st.session_state.current_chat_id = cid
                st.rerun()

            if c2.button("🗑️", key=f"d_{cid}", help="Delete chat"):
                del st.session_state.chat_history[cid]
                if st.session_state.current_chat_id == cid:
                    st.session_state.messages = []
                    st.session_state.current_chat_id = None
                st.rerun()

    st.markdown("---")
    st.markdown("### <i class='fas fa-user'></i> Profile", unsafe_allow_html=True)
    st.markdown("**Name:** Suyash")
    st.markdown(f"**Status:** {'🟢 Connected' if CHATBOT_AVAILABLE else '🟡 Offline Mode'}")
    



# --- Clear chat ---
_, _, col3 = st.columns([5, 5, 1])
with col3:
    st.markdown("""
        <style>
        .clear-chat-button button {
            font-size: 0.75rem !important;
            padding: 0.2rem 0.4rem;
            white-space: nowrap;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.messages and st.button("🧹 Clear", key="clear_button", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

# --- Welcome Section ---
if not st.session_state.messages:
    st.markdown("## Welcome, Suyash!👋")
    st.markdown("### What would you like to explore today?")
    c1, c2, c3 = st.columns(3)
    if c1.button("📈 Help me Analyze Market Trends", use_container_width=True):
        st.session_state.messages.append({"role":"user","content":"Analyze market trends","timestamp":datetime.now()})
        st.rerun()
    if c2.button("🏢 Research my Competitors", use_container_width=True):
        st.session_state.messages.append({"role":"user","content":"Research competitors","timestamp":datetime.now()})
        st.rerun()
    if c3.button("📝 Create a Market Survey", use_container_width=True):
        st.session_state.messages.append({"role":"user","content":"Create survey","timestamp":datetime.now()})
        st.rerun()

# --- Chat Display ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        avatar = "user.png"
        name = "You"
        cls = "user-message"
    else:
        avatar = "TSPlogo.jpg"
        name = "Pragati.TSP"
        cls = "bot-message"

    st.markdown(f"""
    <div class='chat-message {cls}' style='display: flex; align-items: flex-start;'>
        <img src='data:image/png;base64,{get_base64(avatar)}' style='width:40px; height:40px; border-radius:6px; margin-right:10px; object-fit: cover;' />
        <div>
            <strong>{name}:</strong><br>{msg['content']}
        </div>
    </div>
    """, unsafe_allow_html=True)


# # --- File Upload ---
# st.file_uploader("Upload File (optional)", type=["pdf","docx","xlsx","csv","png","jpg"])

# --- Chat Input & Send ---
inp = st.text_area(
    "💬 Ask me anything...",          # Optional icon in label
    height=100,
    placeholder="Type your message here...",  # <-- this is what you wanted
    key="chat_input"  # optional key, helpful for later resets
)

if st.button("📤 Send Message", use_container_width=True):
    if inp.strip():
        usr = {"role": "user", "content": inp, "timestamp": datetime.now()}
        st.session_state.messages.append(usr)

        if not st.session_state.current_chat_id:
            cid = str(uuid.uuid4())
            st.session_state.current_chat_id = cid
            st.session_state.chat_history[cid] = {
                "name": chat_name(inp),
                "messages": [usr]
            }
        else:
            st.session_state.chat_history[st.session_state.current_chat_id]["messages"] = st.session_state.messages

        with st.spinner("Thinking..."):
            resp = bot.get_response(inp).get("response", "") if CHATBOT_AVAILABLE and bot else fallback(inp)
            time.sleep(1)

        st.session_state.messages.append({
            "role": "assistant",
            "content": resp,
            "timestamp": datetime.now()
        })

        st.session_state.chat_history[st.session_state.current_chat_id]["messages"] = st.session_state.messages
        st.rerun()


# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;font-size:0.8rem;'>Powered by Pragati.TSP | Market Research Assistant v1.0</div>", unsafe_allow_html=True)
