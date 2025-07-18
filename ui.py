
import streamlit as st
import requests
import json

st.set_page_config(page_title="Pharmacology Q&A", layout="wide")

st.title("Pharmacology Question Answering System")
st.markdown("This application uses a local vector store and the LLM to answer your questions about pharmacology.")

query = st.text_input("Enter your question:", "")

if st.button("Get Answer"):
    if query:
        with st.spinner("Searching for answers..."):
            try:
                response = requests.post("http://127.0.0.1:5000/query", json={"query": query})
                if response.status_code == 200:
                    st.success("Answer found!")
                    st.write(response.json().get("answer"))
                else:
                    st.error(f"Error from server: {response.text}")
            except requests.exceptions.ConnectionError as e:
                st.error(f"Could not connect to the Flask server. Please make sure it is running. Details: {e}")
    else:
        st.warning("Please enter a question.")
