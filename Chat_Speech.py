import numpy as np
import nltk
import string
import random
import streamlit as st
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the text data
f = open("C:/Users/mudia/OneDrive/Документы/datascienceterms.txt", 'r', errors='ignore')
raw_doc = f.read().lower()
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response):
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sentence_tokens[idx]
    return robo_response

def greet(sentence):
    greet_inputs = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    greet_responses = ["hi", "hey", "whatsup", "hi there", "hello", "I am glad! You are talking to me"]
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)
    return None

def chatbot_response(user_input):
    if greet(user_input) is not None:
        return greet(user_input)
    else:
        return response(user_input)

def transcribe_speech(api_choice, language='en-US'):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio_text = r.listen(source)
        st.info('Transcribing...')
    try:
        if api_choice == 'Google':
            text = r.recognize_google(audio_text, language=language)
        elif api_choice == 'CMU Sphinx':
            text = r.recognize_sphinx(audio_text)
        else:
            text = "Selected API is not supported."
        return text
    except sr.UnknownValueError:
        return "Sorry, I did not understand that. Please try again."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def save_transcription(text):
    with open("transcription.txt", "w") as file:
        file.write(text)
    st.success("Transcription saved to transcription.txt")

def main():
    st.title("Chatbot with Speech Recognition")
    st.write("Type your message or click the microphone to start speaking.")

    user_input = st.text_input("You: ")

    if user_input:
        st.write("Bot: " + chatbot_response(user_input))

    st.write("Or use the speech recognition feature:")

    api_choice = st.selectbox("Select Speech Recognition API", ['Google', 'CMU Sphinx'])
    language = st.selectbox("Select Language", ['en-US', 'es-ES', 'fr-FR', 'de-DE'])

    if st.button("Start Recording"):
        text = transcribe_speech(api_choice, language)
        st.success("Transcription Complete!")
        st.write("Transcription:", text)

        if st.button("Save Transcription"):
            save_transcription(text)

        if text:
            st.write("Bot: " + chatbot_response(text))

if __name__ == "__main__":
    main()
