import streamlit as st
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import uuid
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import chatbot inference
try:
    from chatbot_inference import ChatbotInference
    CHATBOT_AVAILABLE = True
except ImportError as e:
    CHATBOT_AVAILABLE = False
    st.error(f"Failed to import ChatbotInference: {str(e)}. Using fallback responses.")

# Page configuration
st.set_page_config(
    page_title="Pragati.TSP - Market Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background-color: #0A1A2A;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.5rem;
    }
    
    .main-header p {
        color: #cccccc;
        margin: 0;
        font-size: 0.9rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .user-message {
        background-color: #0A1A2A;
        color: white;
        margin-left: 2rem;
    }
    
    .bot-message {
        background-color: #f8f9fa;
        color: #333;
        margin-right: 2rem;
    }
    
    .chat-input {
        border: 2px solid #0A1A2A;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stButton > button {
        background-color: #0A1A2A;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1a2a3a;
    }
    
    .welcome-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .welcome-card:hover {
        border-color: #0A1A2A;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .sidebar-chat {
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        cursor: pointer;
        border: 1px solid transparent;
    }
    
    .sidebar-chat:hover {
        background-color: #f0f0f0;
        border-color: #0A1A2A;
    }
    
    .sidebar-chat.active {
        background-color: #0A1A2A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize chatbot
@st.cache_resource
def initialize_chatbot():
    if CHATBOT_AVAILABLE:
        try:
            return ChatbotInference()
        except Exception as e:
            st.error(f"Failed to initialize ChatbotInference: {str(e)}")
            return None
    return None

chatbot = initialize_chatbot()

# Import and run health check
from health_check import display_health_status
display_health_status()

# Fallback response generator
def generate_fallback_response(user_input: str) -> str:
    input_lower = user_input.lower()
    
    if any(keyword in input_lower for keyword in ['market research', 'market analysis']):
        return """I'd be happy to help with your market research! I can assist with:

• Market size analysis
• Target audience identification  
• Competitive landscape mapping
• Consumer behavior insights
• Industry trend analysis

What specific aspect would you like to explore first?"""
    
    elif any(keyword in input_lower for keyword in ['competitor', 'competition']):
        return """For competitive analysis, I can help you:

• Identify key competitors
• Analyze competitor strategies
• Compare pricing models
• Evaluate market positioning
• Assess competitive advantages

Which competitors or market segment are you focusing on?"""
    
    elif any(keyword in input_lower for keyword in ['survey', 'data']):
        return """I can help you with survey design and data analysis:

• Survey questionnaire development
• Data collection strategies
• Statistical analysis
• Results interpretation
• Actionable insights generation

What type of data are you looking to collect?"""
    
    else:
        return """I understand you're looking for market research assistance. I can help with market analysis, competitive research, consumer insights, and data interpretation. Could you provide more specific details about what you'd like to research?"""

# Generate chat name from message
def generate_chat_name(message: str) -> str:
    keywords = message.lower()
    if any(word in keywords for word in ['market', 'research']):
        return 'Market Research Analysis'
    elif any(word in keywords for word in ['competitor', 'competition']):
        return 'Competitive Analysis'
    elif any(word in keywords for word in ['consumer', 'customer']):
        return 'Consumer Insights Study'
    elif any(word in keywords for word in ['product', 'launch']):
        return 'Product Launch Strategy'
    elif any(word in keywords for word in ['trend', 'analysis']):
        return 'Market Trend Analysis'
    elif any(word in keywords for word in ['survey', 'data']):
        return 'Survey Data Analysis'
    elif any(word in keywords for word in ['brand', 'branding']):
        return 'Brand Research Study'
    else:
        return 'Market Research Chat'

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None

if 'current_chat_name' not in st.session_state:
    st.session_state.current_chat_name = "New Chat"

# Sidebar for chat history
with st.sidebar:
    st.markdown("### 🔍 Explore")
    
    search_term = st.text_input("🔍 Search chats...", placeholder="Search your conversations")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.session_state.current_chat_name = "New Chat"
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.chat_history:
        st.markdown("### 📚 Recent Chats")
        
        filtered_chats = {}
        if search_term:
            filtered_chats = {
                chat_id: chat_data for chat_id, chat_data in st.session_state.chat_history.items()
                if search_term.lower() in chat_data['name'].lower()
            }
        else:
            filtered_chats = st.session_state.chat_history
        
        for chat_id, chat_data in filtered_chats.items():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if st.button(
                    f"💬 {chat_data['name']}", 
                    key=f"chat_{chat_id}",
                    use_container_width=True
                ):
                    st.session_state.messages = chat_data['messages']
                    st.session_state.current_chat_id = chat_id
                    st.session_state.current_chat_name = chat_data['name']
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    del st.session_state.chat_history[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.messages = []
                        st.session_state.current_chat_id = None
                        st.session_state.current_customer_name = "New Chat"
                    st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 👤 User Profile")
    st.markdown("**Name:** Suyash")
    st.markdown(f"**Status:** {'🟢 Connected' if CHATBOT_AVAILABLE else '🟡 Offline Mode'}")

# Main content area
st.markdown("""
<div class="main-header">
    <h1>🤖 Pragati.TSP</h1>
    <p>Market Research Assistant v1.0</p>
</div>
""", unsafe_allow_html=True)

# Clear chat button
col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    if st.session_state.messages and st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.session_state.current_chat_name = "New Chat"
        st.rerun()

# Welcome screen
if not st.session_state.messages:
    st.markdown("## Good Morning, Suyash! 👋")
    st.markdown("### How can I help you today?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Help me analyze market trends", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Help me analyze market trends",
                "timestamp": datetime.now()
            })
            st.rerun()
    
    with col2:
        if st.button("🏢 Research my competitors", use_container_width=True):
            st.session_state.messages.append({
                "role": "user", 
                "content": "Research my competitors",
                "timestamp": datetime.now()
            })
            st.rerun()
    
    with col3:
        if st.button("📋 Create a market survey", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Create a market survey", 
                "timestamp": datetime.now()
            })
            st.rerun()

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Pragati.TSP:</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "📎 Upload a file (optional)", 
    type=['pdf', 'doc', 'docx', 'txt', 'csv', 'xlsx', 'png', 'jpg', 'jpeg'],
    help="Upload documents for analysis"
)

# Chat input
user_input = st.text_area(
    "💬 Ask me anything...", 
    height=100,
    placeholder="Type your message here...",
    key="chat_input"
)

# Send button
if st.button("📤 Send Message", use_container_width=True) and user_input.strip():
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    }
    st.session_state.messages.append(user_message)
    
    if not st.session_state.current_chat_id:
        chat_id = str(uuid.uuid4())
        chat_name = generate_chat_name(user_input)
        
        st.session_state.current_chat_id = chat_id
        st.session_state.current_chat_name = chat_name
        
        st.session_state.chat_history[chat_id] = {
            'name': chat_name,
            'messages': [user_message],
            'timestamp': datetime.now()
        }
    else:
        st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
    
    with st.spinner("🤔 Thinking..."):
        try:
            if CHATBOT_AVAILABLE and chatbot:
                response_data = chatbot.get_response(user_input)
                bot_response = response_data.get('response', 'Sorry, I could not generate a response.')
            else:
                bot_response = generate_fallback_response(user_input)
            
            time.sleep(1)
            
        except Exception as e:
            bot_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
    
    bot_message = {
        "role": "assistant",
        "content": bot_response,
        "timestamp": datetime.now()
    }
    st.session_state.messages.append(bot_message)
    
    if st.session_state.current_chat_id:
        st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
    
    st.rerun()

# Handle file upload
if uploaded_file:
    st.success(f"📎 File uploaded: {uploaded_file.name}")
    st.info("File processing functionality can be integrated with your chatbot_inference.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Powered by Pragati.TSP | Market Research Assistant v1.0
</div>
""", unsafe_allow_html=True)