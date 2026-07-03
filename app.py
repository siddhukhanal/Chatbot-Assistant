import streamlit as st
import os
import json
import random
import torch
import numpy as np

from main import ChatbotAssistant, get_stocks


st.set_page_config(page_title="AI Chatbot Assistant",
                   page_icon="🤖", layout="centered")
st.title("AI Chatbot Assistant")
st.write("Ask me anything! Type `/quit` or clear the chat to start over.")


@st.cache_resource
def load_chatbot():
    """
    Loads and caches the chatbot assistant so it doesn't 
    re-parse or re-load the model on every user interaction.
    """
    assistant = ChatbotAssistant(
        intents_path='intents.json',
        function_mappings={'stocks': get_stocks}
    )

    assistant.parse_intents()

    if os.path.exists('chatbot_mmodel.pth') and os.path.exists('dimensions.json'):
        assistant.load_model('chatbot_mmodel.pth', 'dimensions.json')
    else:
        st.error("Model files not found! Please run your training script first.")

    return assistant


try:
    assistant = load_chatbot()
except Exception as e:
    st.error(f"Failed to initialize chatbot: {e}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input := st.chat_input("Type your message here..."):

    if user_input.strip() == '/quit':
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat session ended. Refresh to restart!"}]
        st.rerun()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = assistant.process_message(user_input)
                if response is None:
                    response = "I'm sorry, I didn't quite catch that. Could you rephrase?"
            except Exception as e:
                response = f"An error occurred while processing your request: {e}"

            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
