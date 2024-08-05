import streamlit as st
import speech_recognition as sr

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
    st.title("Speech Recognition App")
    st.write("Click on the microphone to start speaking")

  
    api_choice = st.selectbox("Select Speech Recognition API", ['Google', 'CMU Sphinx'])

    language = st.selectbox("Select Language", ['en-US', 'es-ES', 'fr-FR', 'de-DE'])

    pause = st.checkbox("Pause/Resume")

    if st.button("Start Recording"):
        if pause:
            st.warning("Recording paused. Uncheck the pause option to resume.")
        else:
            text = transcribe_speech(api_choice, language)
            st.success("Transcription Complete!")
            st.write("Transcription:", text)

            if st.button("Save Transcription"):
                save_transcription(text)

if __name__ == "__main__":
    main()
