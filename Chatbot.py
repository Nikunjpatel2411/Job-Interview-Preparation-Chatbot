import os
import json
import datetime
import random
import nltk
import ssl
import pyttsx3
import speech_recognition as sr
import streamlit as st
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image
import threading

# Setup for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Initialize Translator
translator = Translator()

# Load intents from JSON
file_path = "C:\\Users\\NET\\Downloads\\GPT\\intents.json"
try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except FileNotFoundError:
    st.error("Intents file not found. Please upload it to the correct location.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding the intents JSON file. Please check its format.")
    st.stop()

# Preprocess and Train
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, y)

# Format response
def format_response(intent):
    response = random.choice(intent['responses'])
    if 'additional_info' in intent:
        if 'examples' in intent['additional_info']:
            examples = "\n".join(
                [f"Q: {ex['question']} - A: {ex.get('sample_answer', ex.get('suggestion', ''))}" for ex in
                 intent['additional_info']['examples']])
            response += f"\n\nExamples:\n{examples}"
        if 'resources' in intent['additional_info']:
            resources = "\n".join([f"{res['topic']}: {res['url']}" for res in intent['additional_info']['resources']])
            response += f"\n\nResources:\n{resources}"
    return response

# Chatbot Response
def chatbot(input_text, user_lang):
    try:
        input_text_en = translator.translate(input_text, src=user_lang, dest='en').text
        input_text_vector = vectorizer.transform([input_text_en.lower()])
        predicted_tag = clf.predict(input_text_vector)[0]

        for intent in intents:
            if intent['tag'] == predicted_tag:
                response = format_response(intent)
                # Translate bot's response back to the user's language
                response_translated = translator.translate(response, src='en', dest=user_lang).text
                return response_translated

    except Exception as e:
        return f"I couldn't process your request. Please try again. ({str(e)})"

# Voice Input (Fixing the speech recognition)
recognizer = sr.Recognizer()

def get_audio_input():
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            st.session_state.error_message = ""  # Clear any previous error message
            st.write(f"Recognized: {user_input}")
            return user_input
        except sr.UnknownValueError:
            st.session_state.error_message = "Sorry, I could not understand the audio. Please try again."
            return None
        except sr.RequestError:
            st.session_state.error_message = "Could not request results from Google Speech Recognition service."
            return None

# Text-to-Speech Function with threading
engine = pyttsx3.init()

def speak_response(response):
    def speak():
        engine.say(response)
        engine.runAndWait()

    # Run speech in a separate thread
    speech_thread = threading.Thread(target=speak)
    speech_thread.start()

# Main Application
def main():
    # Sidebar with Menu (Home, Conversation History, About)
    st.sidebar.image("chatbot_logo.png", width=150)
    menu_options = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Menu", menu_options)

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # Home Section
    if choice == "Home":
        # Static Header
        st.markdown("<h1 style='text-align: center;'>Job Interview Preparation Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>Intents-Based Chatbot using NLP</h5>", unsafe_allow_html=True)

        # Sidebar with Language Selection Dropdown
        st.sidebar.title("Language Selection")
        language_options = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "Hindi": "hi",
            "German": "de",
            "Chinese (Simplified)": "zh-cn"
        }
        user_lang = st.sidebar.selectbox("Choose your language:", list(language_options.keys()))
        user_lang_code = language_options[user_lang]

        # Display Chat History (user and bot messages) without timestamps
        for chat in st.session_state.chat_log:
            if chat['sender'] == 'user':
                st.markdown(f"<div style='padding:10px; background-color:#262730; border-radius:10px;color:#edf0eb;margin-bottom:5px;'><B>You:</B> {chat['message']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:10px; background-color:#E5E5E5; border-radius:10px;color:#020500; margin-bottom:5px;'><B>Bot:</B> {chat['message']}</div>", unsafe_allow_html=True)

        # Input Box for User Message and Auto-submit on Enter
        col1, col2 = st.columns([8, 2])
        with col1:
            # Use a unique key for the user input field
            user_input = st.text_input("Type your message or click the mic to speak:", key="user_input_home", label_visibility="collapsed")
        with col2:
            if st.button("Speak", use_container_width=True):
                recognized_text = get_audio_input()
                if recognized_text:
                    st.session_state.recognized_text = recognized_text

        # Update the input field with recognized text from speech
        if "recognized_text" in st.session_state and st.session_state.recognized_text:
            user_input = st.session_state.recognized_text

        # Display the error message if it's set in session state
        if "error_message" in st.session_state and st.session_state.error_message:
            st.error(st.session_state.error_message)

        # If the user presses Enter or submits input
        if user_input:
            # Append user input to chat log without timestamp for Home section
            st.session_state.chat_log.append({"sender": "user", "message": user_input, "timestamp": datetime.datetime.now()})

            # Get chatbot response
            response = chatbot(user_input, user_lang_code)

            # Append bot response to chat log with timestamp for Home section
            st.session_state.chat_log.append({"sender": "bot", "message": response, "timestamp": datetime.datetime.now()})

            # Text-to-Speech
            speak_response(response)

            # Instead of clearing `st.session_state.user_input`, we reset a custom variable
            st.session_state.recognized_text = ""  # Clear recognized text after submission

    # Conversation History Section
    elif choice == "Conversation History":
        st.title("Conversation History")
        if len(st.session_state.chat_log) == 0:
            st.write("No conversation history yet.")
        else:
            # Display conversation with timestamp in History section
            for chat in st.session_state.chat_log:
                timestamp = chat['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                if chat['sender'] == 'user':
                    st.markdown(f"You ({timestamp}): {chat['message']}")
                else:
                    st.markdown(f"Bot ({timestamp}): {chat['message']}")

    # About Section
    elif choice == "About":
        st.title("About the Chatbot")
        st.write("""
        This chatbot assists users in preparing for job interviews by providing tips, answering common questions, 
        and offering advice. It utilizes NLP, machine learning, and multilingual support for an interactive experience.
        """)

if __name__ == '__main__':
    main()
