import streamlit as st
import requests

st.title("Code Security Checker")

code = st.text_area("Enter your code:")
if st.button("Check Code"):
    response = requests.post("http://127.0.0.1:8000/predict/", json={"code": code})
    result = response.json().get("result", "Error")
    st.success(f"Result: {result}")
