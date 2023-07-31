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


def gather_corpus(path: str, final_file_name: str) -> pd.DataFrame:
    st = time.time()
    # only emails from these senders are to be processed
    users_emails = set(pd.read_csv("emails.csv")["email"])
    email_translator_df = pd.read_csv("emails_translator.csv")
    emails_to_translate = set(email_translator_df["email"])

    # loads emails from dataset
    email_list = []
    load_from_dir(path, email_list, users_emails, emails_to_translate, email_translator_df)

    # saves emails to csv
    email_df = pd.DataFrame(email_list, columns=['sender', 'text', 'path'])
    email_df[["sender", "text"]].duplicated(keep="first")
    email_df.to_csv(final_file_name)
    print(email_df[["path", "text"]])
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    return email_df


def load_from_dir(path: str, email_list: list[list[str]], user_addresses: set[str], translatable_addresses: set[str]
                  , email_translator: pd.DataFrame) -> None:
    print(path)
    load_emails(path, email_list, user_addresses, translatable_addresses, email_translator)
    dir_list = os.listdir(path)
    dirs = [f for f in dir_list if os.path.isdir(path + '/' + f)]
    for dir_name in dirs:
        load_from_dir(path + '/' + dir_name, email_list, user_addresses, translatable_addresses, email_translator)


def load_emails(path: str, email_list: list[list[str]], user_addresses: set[str], translatable_addresses: set[str]
                  , email_translator: pd.DataFrame) -> None:
    if "to_do" in path: # asi nějaký automatický emaily generovaný
        return
        # možná i calendar by se měl vyfiltrovat?
    dir_list = os.listdir(path)
    files = [f for f in dir_list if os.path.isfile(path + '/' + f)]
    for file in files:
        try:
            load_email(path + '/' + file, email_list, user_addresses, translatable_addresses, email_translator)
        except UnicodeDecodeError:
            load_email(path + '/' + file, email_list, user_addresses, translatable_addresses, email_translator, code="latin-1")


def load_email(file_path: str, email_list: list[list[str]], user_addresses: set[str], translatable_addresses: set[str]
                  , email_translator: pd.DataFrame, code="utf-8") -> None:
    with open(file_path, 'r', encoding=code) as file_desc:
        text = file_desc.read()
        if "--- Forwarded by " in text or "-----Original Message" in text or "-----Original Appointment" in text:
            return
            # potom předělat na jenom slice po klíčový slova
        sender_index_start = text.find("From: ")
        if sender_index_start == -1:
            raise Exception("Sender is missing")
        sender = text[sender_index_start + 6: text.find("\n", sender_index_start)]
        if sender in user_addresses:
            email_text = process_text(text)
            email_list.append([sender, email_text, file_path])
        elif sender in translatable_addresses:
            translate_to = email_translator[email_translator["email"] == sender]["main_email"].values[0]
            email_text = process_text(text)
            email_list.append([translate_to, email_text, file_path])


def gather_user_emails():
    path = "enron_mail/maildir"
    dir_list = os.listdir(path)
    dirs = [f for f in dir_list if os.path.isdir(path + '/' + f)]
    address_final = []
    address_translator = []
    for dir_name in dirs:
        print(dir_name)
        emails = []
        address = set()
        load_from_dir_address(path + '/' + dir_name, emails, address)
        checked_addresses = []
        if len(address) > 0:
            for email in address:
                if email and dir_name.split("-")[0] in email:
                    checked_addresses.append(email)
        email_df = pd.DataFrame(emails, columns=['sender', 'text', 'path'])
        if len(checked_addresses) > 0:
            if len(checked_addresses) > 1:
                emails_ordered = email_df.groupby("sender").count().sort_values(by="text", ascending=False)
                main_address = emails_ordered.index[0]
                address_final.append(main_address)
                print(main_address)
                for address in checked_addresses:
                    if address != main_address:
                        address_translator.append([address, main_address])
            else:
                address_final.append(checked_addresses[0])
                print(checked_addresses[0])
        print("")

    print(address_final)
    print(address_translator)
    print(len(address_final))
    print(len(address_translator))
    emails_df = pd.DataFrame(address_final, columns=['email'])
    emails_df.to_csv("emails.csv")
    emails_df = pd.DataFrame(address_translator, columns=['email', 'main_email'])
    emails_df.to_csv("emails_translator.csv")


def load_from_dir_address(path: str, email_list: list[list[str]], address: set[str]) -> None:
    load_emails_address(path, email_list, address)
    dir_list = os.listdir(path)
    dirs = [f for f in dir_list if os.path.isdir(path + '/' + f)]
    for dir_name in dirs:
        load_from_dir_address(path + '/' + dir_name, email_list, address)


def load_emails_address(path: str, email_list: list[list[str]], address: set[str]) -> None:
    if not "sent" in path:
        return
    dir_list = os.listdir(path)
    files = [f for f in dir_list if os.path.isfile(path + '/' + f)]
    for file in files:
        try:
            address.add(load_email_address(path + '/' + file, email_list, address))
        except UnicodeDecodeError:
            address.add(load_email_address(path + '/' + file, email_list, address, code="latin1"))


def load_email_address(file_path: str, email_list: list[list[str]], address: set[str], code="utf-8") -> str:
    with open(file_path, 'r', encoding=code) as file_desc:
        text = file_desc.read()
        # if "--- Forwarded by " in text or "-----Original Message" in text or "-----Original Appointment" in text:
        #     return
        sender_index_start = text.find("From: ")
        if sender_index_start == -1:
            raise Exception("Sender is missing")
        sender = text[sender_index_start + 6: text.find("\n", sender_index_start)]
        if sender == 'no.address@enron.com':
            return
        if sender == 'brenda.whitehead@enron.com':
            pass
        email_text = process_text(text)
        email_list.append([sender, email_text, file_path])
        return sender


if __name__ == "__main__":
    gather_user_emails()
    gather_corpus("enron_mail", "corpus.csv")
    # data = pd.read_csv("corpus.csv", index_col=0)


    # check proč gather addresses nevzalo rodrigue

