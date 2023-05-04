from app.core.core import predict_answer
from app.tools.tts import tts

import speech_recognition as sr
from time import sleep

r = sr.Recognizer()

close = ["bye", "goodbye", "exit", "quit", "close"]

while True:
    with sr.Microphone() as source:
        welcome = "How can I help you?"

        print(f"\n{welcome}")
        tts(welcome)

        audio = r.listen(source)

    try:
        print("Proccessing . . .")

        text = r.recognize_google(audio)

        if (text in close):
            bye = "Glad to talk"

            print(bye)
            tts(bye)

            exit()

        prediction = predict_answer(text)

        print(f"Prompt is: {text}")
        print(f"Your answer: {prediction}")

        tts(prediction)
    except sr.UnknownValueError:
        print("Could not understand audio")
        tts("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results")
        tts("Could not request results")

    sleep(1)
