# Pragati.TSP - Market Research Assistant

Pragati.TSP is a Streamlit-based chatbot for market research, leveraging MongoDB for real-time data storage and the Groq API for enhanced data extraction.

## Project Structure

PragatiTSP-Chatbot/
├── app.py
├── chatbot_inference.py
├── models/
│   └── PragatiTSP_components.pkl
├── requirements.txt
├── health_check.py
├── .streamlit/
│   └── config.toml
├── .env
├── README.md
├── .gitignore



## Prerequisites
- Python 3.8+
- MongoDB Atlas account
- Groq API key
- Streamlit Community Cloud account
- GitHub account

## Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/PragatiTSP-Chatbot.git
   cd PragatiTSP-Chatbot
2. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
3. Create a .env file with your credentials:
   GROQ_API_KEY=your_groq_api_key_here
   MONGO_URI=mongodb+srv://your_username:your_password@your_cluster.mongodb.net/
4. Run the app locally:
   streamlit run app.py 