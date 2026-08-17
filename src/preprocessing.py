# use RE for text cleaning
import re

def clean_text(text):
    # convert text to lowercase
    text = text.lower()
    # remove URLs 
    text = re.sub(r"http\S+|www\S+", " ", text)
    # remove  XXXX or XXXXXXXXX
    text = re.sub(r"\bx+\b", " ", text)
    # keep letters, numbers, and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # replace multiple spaces
    text = re.sub(r"\s+", " ", text)
    # remove spaces from the beginning and end
    text = text.strip()

    return text


def clean_texts(texts):
    # Apply clean_text() to every complaint
    return [clean_text(text) for text in texts]