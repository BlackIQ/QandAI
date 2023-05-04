from os import system


def tts(text):
    system(f"say {text}")
