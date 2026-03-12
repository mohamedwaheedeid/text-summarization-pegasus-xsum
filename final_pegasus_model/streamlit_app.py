# streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="Pegasus Summarizer", page_icon="✂️", layout="wide")
st.title("Pegasus Text Summarizer")

st.write("Enter text below and click 'Summarize' to get a summary.")

# Text input box
user_text = st.text_area("Enter your text here:", height=200)

if st.button("Summarize"):
    if user_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        try:
            # Call the FastAPI endpoint
            url = "http://127.0.0.1:8000/summarize"
            response = requests.post(url, json={"text": user_text})
            summary = response.json().get("summary", "")
            
            if summary:
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.error("No summary returned. Check your API.")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")