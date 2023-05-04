from os import system


def tts(text):
    system(f"say {text}")


tts("It's me! A machine talking!!")
