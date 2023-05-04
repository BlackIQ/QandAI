import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Speak:")
    audio = r.listen(source)

try:
    text = r.recognize_google(audio)
    print(f"You said, {text}")
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request")
