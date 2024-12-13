import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load the model and tokenizer globally to avoid reloading on each request
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

# Streamlit app UI
st.title("English to French Translator")

# Input field for user to enter text to translate
text = st.text_area("Enter text to translate:")

if st.button("Translate"):
    if text:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Perform translation
        with torch.no_grad():  # Turn off gradients for inference
            translated_tokens = model.generate(**inputs)
        
        # Decode the translated tokens to text
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # Display translated text
        st.write("Translated Text: ", translated_text)
    else:
        st.warning("Please enter text to translate.")
