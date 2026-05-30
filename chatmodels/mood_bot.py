import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

st.set_page_config(page_title="Mistral Mood Chatbot", page_icon="🤖", layout="centered")

@st.cache_resource
def initialize_chatbot():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Error: MISTRAL_API_KEY not found in the environment setup.")
        st.stop()
        
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.9,
        mistral_api_key=api_key
    )

try:
    model = initialize_chatbot()
except Exception as e:
    st.error(f"Failed to start chatbot: {e}")
    st.stop()

st.sidebar.title("⚙️ Configuration")
mood_choice = st.sidebar.selectbox(
    "Select Assistant Mood:",
    ["Funny", "Angry", "Sad", "Romantic", "Sarcastic", "Custom"]
)

if mood_choice == "Custom":
    selected_mood = st.sidebar.text_input("Type your custom mood:", value="helpful")
else:
    mood_mapping = {
        "Funny": "funny assistant who cracks jokes often",
        "Angry": "very angry assistant who gets annoyed quickly",
        "Sad": "sad and depressed assistant",
        "Romantic": "romantic and poetic assistant",
        "Sarcastic": "sarcastic assistant who makes fun of everything"
    }
    selected_mood = mood_mapping[mood_choice]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {mood}."),
])
formatted_prompt = prompt_template.format_prompt(mood=selected_mood)
system_message = formatted_prompt.to_messages()[0]

st.title("🤖 Mistral AI Mood Chatbot")
st.caption(f"Current Persona: **{mood_choice if mood_choice != 'Custom' else selected_mood}**")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

if user_query := st.chat_input("Say something to the bot..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    full_messages_payload = [system_message] + st.session_state.chat_history

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = model.invoke(full_messages_payload)
                st.write(response.content)
                st.session_state.chat_history.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"API Error: {e}")