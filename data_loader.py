import os
import time

import pandas as pd


def load_corpus(path):
    corpus = {}
    dir_list = os.listdir(path)
    files = [f for f in dir_list if os.path.isfile(path + '/' + f)]
    for file in files:
        name = file.split('.txt')[0]
        load_email({}, path + '/' + file)
        with open(path + '/' + file, 'r', encoding='utf-8') as f:
            text = f.read()
        corpus[name] = text
    return corpus


def process_text(full_email_text: str) -> str:
    by_enter = full_email_text.split('\n')
    if len(by_enter) < 15:
        raise Exception("Header is missing")
    email_body_start = 15
    if "X-FileName:" in by_enter[14]:
        pass
    else:
        for line_number, line in enumerate(by_enter):
            if "X-FileName:" in line:
                email_body_start = line_number + 1
                break
    email_text = "\n".join(by_enter[email_body_start:])
    return email_text


def gather_corpus(path: str) -> pd.DataFrame:
    st = time.time()
    email_list = []
    load_from_dir(path, email_list)
    email_df = pd.DataFrame(email_list, columns=['sender', 'text', 'path'])
    email_df[["sender", "text"]].duplicated(keep="first")
    email_df.to_csv("corpus.csv")
    print(email_df[["path", "text"]])
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    return email_df


def load_from_dir(path: str, email_list: list[list[str]]) -> None:
    load_emails(path, email_list)
    dir_list = os.listdir(path)
    dirs = [f for f in dir_list if os.path.isdir(path + '/' + f)]
    for dir_name in dirs:
        print(path + '/' + dir_name)
        load_from_dir(path + '/' + dir_name, email_list)


def load_emails(path: str, email_list: list[list[str]]) -> None:
    dir_list = os.listdir(path)
    files = [f for f in dir_list if os.path.isfile(path + '/' + f)]
    for file in files:
        try:
            load_email(path + '/' + file, email_list)
        except UnicodeDecodeError:
            load_email(path + '/' + file, email_list, code="latin1")


def load_email(file_path: str, email_list: list[list[str]], code="utf-8") -> None:
    with open(file_path, 'r', encoding=code) as file_desc:
        text = file_desc.read()
        if "--- Forwarded by " in text or "-----Original Message-----" in text:
            return
        sender_index_start = text.find("From: ")
        if sender_index_start == -1:
            raise Exception("Sender is missing")
        sender = text[sender_index_start + 6: text.find("\n", sender_index_start)]
        email_text = process_text(text)
        email_list.append([sender, email_text, file_path])

