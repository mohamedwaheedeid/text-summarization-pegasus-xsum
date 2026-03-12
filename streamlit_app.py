import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Pegasus AI Summarizer", 
    page_icon="📝", 
    layout="centered"
)

# 2. Custom CSS for Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .summary-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1.1em;
        line-height: 1.6;
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar Configuration
with st.sidebar:
    st.title("🤖 Model Info")
    st.info("**Google Pegasus-XSum**")
    st.write("""
    This model is specialized in **Abstractive Summarization**. 
    It rewrites content to create concise, human-like summaries.
    """)
    st.markdown("---")
    
    # Connection Check
    try:
        check = requests.get("http://127.0.0.1:8000/", timeout=2)
        status = "API Connected ✅" if check.status_code == 200 else "API Error ⚠️"
    except:
        status = "API Offline ❌"
    
    st.caption(f"Status: {status}")

# 4. Main Interface
st.title("📝 AI Text Summarizer")
st.markdown("Transform long articles into intelligent summaries instantly.")

# Text Input Area
user_text = st.text_area(
    "📄 Paste your text here:", 
    height=250, 
    placeholder="Type or paste your content here..."
)

# Summarization Logic
if st.button("✨ Generate Summary"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("🤖 Processing... This may take a few seconds"):
            try:
                # API Call to your FastAPI server
                url = "http://127.0.0.1:8000/summarize"
                response = requests.post(url, json={"text": user_text}, timeout=120)
                
                if response.status_code == 200:
                    summary = response.json().get("summary", "")
                    
                    if summary:
                        st.success("✅ Summary Generated!")
                        st.markdown("### 🎯 Result")
                        
                        # Display result in the styled box
                        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                        
                        # Metrics
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Words", len(user_text.split()))
                        with col2:
                            st.metric("Summary Words", len(summary.split()))
                    else:
                        st.error("❌ The model returned an empty result.")
                else:
                    st.error(f"❌ Server Error: {response.status_code}")

            except Exception as e:
                st.error("📡 Connection Failed! Is the FastAPI server running?")
                st.info("Ensure you have run: `uvicorn app:app --host 127.0.0.1 --port 8000` in your other terminal.")

# 5. Footer
st.markdown("---")
st.caption("Powered by Pegasus AI & Streamlit")