import streamlit as st
from datetime import datetime
import sys
import platform

def get_system_info():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "streamlit_version": st.__version__,
        "uptime": "Available during session"
    }

def check_chatbot_health():
    try:
        from chatbot_inference import ChatbotInference
        chatbot = ChatbotInference()
        health_status = chatbot.health_check()
        return {
            "chatbot_status": "available",
            "chatbot_health": health_status
        }
    except ImportError:
        return {
            "chatbot_status": "fallback_mode",
            "message": "Using fallback responses - chatbot_inference.py not available"
        }
    except Exception as e:
        return {
            "chatbot_status": "error",
            "error": str(e)
        }

def display_health_status():
    system_info = get_system_info()
    chatbot_info = check_chatbot_health()
    
    with st.sidebar.expander("🔧 System Status"):
        st.json({
            **system_info,
            **chatbot_info
        })

if __name__ == "__main__":
    print("System Info:", get_system_info())
    print("Chatbot Info:", check_chatbot_health())