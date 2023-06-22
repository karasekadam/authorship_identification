import re
from nltk.corpus import stopwords
import pandas as pd
import time


def num_of_words(text: str) -> int:
    return len(re.findall(r'\w+', text))


def num_of_sentences(text: str) -> int:
    return len(re.findall(r'\.', text))


def num_of_lines(text: str) -> int:
    return len(re.findall(r'\n', text))


def num_of_uppercase(text: str) -> int:
    return len(re.findall(r'[A-Z]', text))


def num_of_titlecase(text: str) -> int:
    return len(re.findall(r'[A-Z][a-z]', text))


def average_len_of_words(text: str) -> float:
    words = re.findall(r'\w+', text)
    if len(words) == 0:
        return 0
    return sum([len(word) for word in words]) / len(words)


def num_of_punctuation(text: str) -> int:
    return len(re.findall(r'[.,!?]', text))


def num_of_special_chars(text: str) -> int:
    return len(re.findall(r'[^A-Za-z0-9]', text))


def num_of_chars(text: str) -> int:
    return len(text)


def num_of_stopwords(text: str) -> int:
    return len(re.findall(r'\b(?:{})\b'.format('|'.join(stopwords.words('english'))), text))


def num_of_unique_words(text: str) -> int:
    return len(set(re.findall(r'\w+', text)))


# def num_of_unique_words_ratio(text: str) -> float:
#    return num_of_unique_words(text) / num_of_words(text)


def num_of_digits(text: str) -> int:
    return len(re.findall(r'\d', text))


#def num_of_digits_ratio(text: str) -> float:
#    return num_of_digits(text) / num_of_chars(text)


def calculate_stylometry(df: pd.DataFrame) -> None:
    print(time.time())
    df["num_of_words"] = df["text"].apply(num_of_words)
    print(time.time())
    df["num_of_sentences"] = df["text"].apply(num_of_sentences)
    print(time.time())
    df["num_of_lines"] = df["text"].apply(num_of_lines)
    print(time.time())
    df["num_of_uppercase"] = df["text"].apply(num_of_uppercase)
    print(time.time())
    df["num_of_titlecase"] = df["text"].apply(num_of_titlecase)
    print(time.time())
    df["average_len_of_words"] = df["text"].apply(average_len_of_words)
    print(time.time())
    df["num_of_punctuation"] = df["text"].apply(num_of_punctuation)
    print(time.time())
    df["num_of_special_chars"] = df["text"].apply(num_of_special_chars)
    print(time.time())
    df["num_of_chars"] = df["text"].apply(num_of_chars)
    print(time.time())
    df["num_of_stopwords"] = df["text"].apply(num_of_stopwords)
    print(time.time())
    df["num_of_unique_words"] = df["text"].apply(num_of_unique_words)
    print(time.time())
    df["num_of_digits"] = df["text"].apply(num_of_digits)
    print(time.time())
    df.drop(columns=["text"], inplace=True)

