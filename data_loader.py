import os
import time
import re
import pandas as pd
from header_columns import header_metadata_columns


def process_text(full_email_text: str) -> str | None:
    by_enter = full_email_text.split('\n')
    if len(by_enter) < 15:
        return
        # raise Exception("Header is missing")
    email_body_start = 15
    if "X-FileName:" in by_enter[14]:
        pass
    else:
        for line_number, line in enumerate(by_enter):
            if "X-FileName:" in line:
                email_body_start = line_number + 1
                break
    email_text = "\n".join(by_enter[email_body_start:])

    if email_text.count("synchronizing") > 10 or email_text.count("Synchronizing"):
        return

    return email_text


def gather_corpus(path: str, final_file_name: str) -> pd.DataFrame:
    st = time.time()
    # only emails from these senders are to be processed
    users_emails = set(pd.read_csv("emails.csv")["email"])
    email_translator_df = pd.read_csv("emails_translator.csv", index_col=0)
    emails_to_translate = set(email_translator_df["email"])

    # loads emails from dataset
    email_list = []
    load_from_dir(path, email_list, users_emails, emails_to_translate, email_translator_df)

    # saves emails to csv
    email_df = pd.DataFrame(email_list, columns=['author', 'text', 'path'] + header_metadata_columns)
    email_df[["author", "text"]].duplicated(keep="first")
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
    if "to_do" in path or "contacts" in path or "calendar" in path:  # asi nějaký automatický emaily generovaný
        return
        # možná i calendar by se měl vyfiltrovat?
    dir_list = os.listdir(path)
    files = [f for f in dir_list if os.path.isfile(path + '/' + f)]
    for file in files:
        try:
            load_email(path + '/' + file, email_list, user_addresses, translatable_addresses, email_translator)
        except UnicodeDecodeError:
            load_email(path + '/' + file, email_list, user_addresses, translatable_addresses, email_translator, code="latin-1")


key_words_for_slicing = ["--- Forwarded by", "--- Original Message", "--- Original Appointment", "---Forwarded by",
                         "---Original Message", "---Original Appointment"]


def slice_text(text: str, file_path: str) -> str:
    lowest_position = None
    for key_word in key_words_for_slicing:
        search_end = len(text) if lowest_position is None else lowest_position
        index = text.find(key_word, 0, search_end)
        if index != -1:
            if lowest_position is None or index < lowest_position:
                lowest_position = index

    position_to_slice = lowest_position
    if position_to_slice is not None:
        while position_to_slice > 0 and text[position_to_slice] == "-":
            position_to_slice -= 1
        text = text[:position_to_slice]

    if text.count("\n\n\n\n\n\n") > 0:
        res = [i.start() for i in re.finditer("\n\n\n\n\n\n", text)]
        text = text[:res[0]]

    if text.count("To:") > 2:
        res = [i.start() for i in re.finditer("To:", text)]
        text = text[:res[2]]
    if text.count("From:") > 2:
        res = [i.start() for i in re.finditer("From:", text)]
        text = text[:res[2]]
    if text.count("Sent by:") > 2:
        res = [i.start() for i in re.finditer("Sent by:", text)]
        text = text[:res[2]]

    for author in signature:
        if author in file_path:
            for signature_type in signature[author]:
                if len(signature_type) <= 2:
                    email_start = text[:-(10 + len(signature_type))]
                    email_end = text[-(10 + len(signature_type)):]
                    email_end = email_end.replace(signature_type, "")
                    text = email_start + email_end
                else:
                    email_start = text[:-(200 + len(signature_type))]
                    email_end = text[-(200 + len(signature_type)):]
                    email_end = email_end.replace(signature_type, "")
                    text = email_start + email_end

    return text


def check_existing_text(text: str) -> bool:
    if text is None:
        return False
    check_text = text.replace(" ", "")
    check_text = check_text.replace("\n", "")
    check_text = check_text.replace("\t", "")
    if len(check_text) < 1:
        return False
    return True


def email_subject(text: str) -> list[int]:
    subject_index_start = text.find("Subject: ")
    if subject_index_start == -1:
        return [0, 0, 0, 0, 0]

    subject_index_end_mime = text.find("Mime-Version:", subject_index_start)
    subject_index_end_cc = text.find("Cc:", subject_index_start)
    if subject_index_end_mime == -1 and subject_index_end_cc == -1:
        return [0, 0, 0, 0, 0]
    elif subject_index_end_mime == -1:
        subject_index_end = subject_index_end_cc
    elif subject_index_end_cc == -1:
        subject_index_end = subject_index_end_mime
    else:
        subject_index_end = min(subject_index_end_mime, subject_index_end_cc)

    subject = text[subject_index_start + 9: subject_index_end]
    number_of_words = len(subject.split(" "))
    number_of_characters = len(subject)
    number_of_uppercase_characters = sum(1 for c in subject if c.isupper())
    number_od_numeric_characters = sum(1 for c in subject if c.isnumeric())
    number_of_punctuation_marks = sum(1 for c in subject if c in [".", ",", "!", "?", ";", ":", "-", "_", "(", ")"])

    return [number_of_words, number_of_characters, number_of_uppercase_characters, number_od_numeric_characters,
            number_of_punctuation_marks]


def email_addressee(text: str, author_email: str) -> list[int]:
    addressee_metadata = [0, 0, 0, 0]

    addressee_index_start = text.find("To: ")
    addressee_index_end = text.find("Subject:", addressee_index_start)
    if addressee_index_end != -1 and addressee_index_start != -1:
        addressee_text = text[addressee_index_start + 4: addressee_index_end]
        addressee = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', addressee_text)
        addressee_metadata[0] = len(addressee)
        addressee_metadata[1] = sum([1 for email in addressee if author_email.split("@")[1] in email])

    cc_index_start = text.find("Cc: ")
    cc_index_end = text.find("Mime-Version:", cc_index_start)
    if cc_index_end != -1 and cc_index_start != -1:
        cc_text = text[cc_index_start + 4: cc_index_end]
        cc = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', cc_text)
        addressee_metadata[2] = len(cc)
        addressee_metadata[3] = sum([1 for email in cc if author_email.split("@")[1] in email])

    return addressee_metadata


def process_time(text: str) -> int:
    date_index_start = text.find("Date: ")
    time_text = text[date_index_start + 23: date_index_start + 25].replace(":", "")
    return int(time_text)


def load_email(file_path: str, email_list: list[list[str]], user_addresses: set[str], translatable_addresses: set[str],
               email_translator: pd.DataFrame, code="utf-8") -> None:
    with open(file_path, 'r', encoding=code) as file_desc:
        if "kaminski-v/sent_items/738" in file_path:
            pass

        text = file_desc.read()

        # gets sender
        sender_index_start = text.find("From: ")
        if sender_index_start == -1:
            raise Exception("Sender is missing")
        sender = text[sender_index_start + 6: text.find("\n", sender_index_start)]
        if "@" not in sender:
            return

        if sender not in user_addresses:
            translation_email = email_translator[email_translator["email"] == sender]["main_email"]
            if len(translation_email) == 0:
                return
            sender = translation_email.values[0]

        text = slice_text(text, file_path)

        # process time
        time = process_time(text)

        # process subject
        subject_data = email_subject(text)

        # process addressee
        addressee_data = email_addressee(text, sender)

        # gets email text
        email_text = process_text(text)
        if not check_existing_text(email_text):
            return

        if file_path == "enron_mail/maildir/beck-s/calendar/120":
            pass
        email_list.append([sender, email_text, file_path, time] + subject_data + addressee_data)


def gather_user_emails():
    path = "enron_mail/maildir"
    dir_list = os.listdir(path)
    dirs = [f for f in dir_list if os.path.isdir(path + '/' + f)]
    address_final = []
    address_translator = []
    for dir_name in dirs:
        emails = []
        address = set()
        load_from_dir_address(path + '/' + dir_name, emails, address)
        checked_addresses = []
        if len(address) > 0:
            for email in address:
                if email and dir_name.split("-")[0] in email:
                    checked_addresses.append(email)
        email_df = pd.DataFrame(emails, columns=['author'])
        if len(checked_addresses) > 0:
            if len(checked_addresses) > 1:
                emails_ordered = email_df.groupby("author").size().sort_values(ascending=False)
                main_address = emails_ordered.index[0]
                address_final.append(main_address)
                for address in checked_addresses:
                    if address != main_address:
                        address_translator.append([address, main_address])
            else:
                address_final.append(checked_addresses[0])

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


#
def load_email_address(file_path: str, email_list: list[list[str]], address: set[str], code="utf-8") -> str | None:
    with open(file_path, 'r', encoding=code) as file_desc:
        text = file_desc.read()
        sender_index_start = text.find("From: ")
        if sender_index_start == -1:
            raise Exception("Sender is missing")

        sender = text[sender_index_start + 6: text.find("\n", sender_index_start)]
        if sender == 'no.address@enron.com':
            return
        if sender == 'brenda.whitehead@enron.com':
            pass
        email_list.append([sender])
        return sender


# filters only n most active senders
def filter_most_active_authors(n: int, df: pd.DataFrame) -> pd.DataFrame:
    # data = pd.read_csv("corpus.csv", index_col=0)
    data_grouped = df.groupby("author").size().sort_values(ascending=False)

    first_n_emails = list(data_grouped.head(n).index)
    data = df[df['author'].isin(first_n_emails)]
    data = data.reset_index(drop=True)
    return data
    # data.to_csv("corpus" + str(n) + ".csv")


def process_techcrunch():
    df = pd.read_csv("techcrunch_posts.csv")
    df = df.rename(columns={"authors": "author", "content": "text"})
    df.to_csv("techcrunch.csv")


def process_telegram():
    # zdroj https://www.kaggle.com/datasets/aagghh/crypto-telegram-groups?select=group_messages_okex.json
    df = pd.read_json("group_messages_okex.json")
    df = df[["message", "from_id"]]
    df["message"] = df["message"].astype(str)
    df = df[df["message"].apply(lambda x: len(x) > 50)]

    df = df[df["from_id"].apply(lambda x: "PeerUser" in x.values())]
    dict_values = df["from_id"].apply(lambda x: list(x.values()))
    df_dict_values = pd.DataFrame(dict_values.tolist())
    df = df.merge(df_dict_values, left_index=True, right_index=True)
    df = df.drop(columns=["from_id", 0])
    df = df.rename(columns={1: "author", "message": "text"})
    return df


def prepare_experiment_set(number_of_authors: int, samples_per_author: int, data: pd.DataFrame, output_name: str) -> None:
    data["text"] = data["text"].astype(str)
    data = data[data["text"].apply(lambda x: len(x) > 100)]
    data = filter_most_active_authors(number_of_authors, data)
    authors = data["author"].unique()
    test_sample = pd.DataFrame(columns=data.columns)
    for author in authors:
        author_data = data[data["author"] == author]
        author_sample = author_data.sample(n=samples_per_author)
        test_sample = pd.concat([test_sample, author_sample], ignore_index=True)

    test_sample = test_sample.reset_index(drop=True)
    test_sample.to_csv(output_name)


def prepare_all_experiments_sets():
    # 5 authors experiment sets
    df_enron = pd.read_csv("enron.csv", index_col=0)
    prepare_experiment_set(5, 4000, df_enron, "experiment_sets/enron_experiment_sample_5.csv")

    df_techcrunch = pd.read_csv("techcrunch.csv", index_col=0)
    prepare_experiment_set(5, 2500, df_techcrunch, "experiment_sets/techcrunch_experiment_sample_5.csv")

    df_telegram = pd.read_csv("telegram.csv", index_col=0)
    prepare_experiment_set(5, 1000, df_telegram, "experiment_sets/telegram_experiment_sample_5.csv")

    # 10 authors experiment sets
    df_enron = pd.read_csv("enron.csv", index_col=0)
    prepare_experiment_set(10, 1800, df_enron, "experiment_sets/enron_experiment_sample_10.csv")

    df_techcrunch = pd.read_csv("techcrunch.csv", index_col=0)
    prepare_experiment_set(10, 1200, df_techcrunch, "experiment_sets/techcrunch_experiment_sample_10.csv")

    df_telegram = pd.read_csv("telegram.csv", index_col=0)
    prepare_experiment_set(10, 650, df_telegram, "experiment_sets/telegram_experiment_sample_10.csv")

    # 25 authors experiment sets
    df_enron = pd.read_csv("enron.csv", index_col=0)
    prepare_experiment_set(25, 700, df_enron, "experiment_sets/enron_experiment_sample_25.csv")

    df_techcrunch = pd.read_csv("techcrunch.csv", index_col=0)
    prepare_experiment_set(25, 250, df_techcrunch, "experiment_sets/techcrunch_experiment_sample_25.csv")

    df_telegram = pd.read_csv("telegram.csv", index_col=0)
    prepare_experiment_set(25, 470, df_telegram, "experiment_sets/telegram_experiment_sample_25.csv")


signature = {
    "kaminski-v": ["Vince", "Kaminski"],
    "dasovich-j": ["Jeff", "jeff"],
    "germany-c": ["Chris", "Germany"],
    "mann-k": ["Kay", "kay"],
    "jones-t": ["Tana"],
    "beck-s": ["--Sally"],
    "delainey-d": ["Delainey", "Dave"],
    "lavorato-j": ["John", "Lavo"],
    "mcconnell-m": ["Mike", "m", "mike"],
    "nckaughkub-e": ["Errol", "McLaughlin"],
    "gigor-d": ["DG", "dg", "Dg"],
    "arnorld-j": ["John", "john"],
    "fossum-d": ["DF", "df", "Df"],
    "allen-p": ["Phillip", "Allen"],
    "cash-m": ["Michelle", "Cash"],
    "grigsby-m": ["Mike", "Grigsby", "Grigs"],
    "haedicke-m": ["Mark", "Haedicke"],
    "lenhart-m": ["Matt", "matt"],
    "love-p": ["PL", "pl"],
    "farmer-d": ["D", "Daren", "Farmer"],
    "bass-e": ["-Eric", "-e", "Eric", "-E", "e"],
    "campbell-l": ["LC", "Larry", "Campbell"],
    "dorland-c": ["Chris", "CD", "Cd", "cd"],
    "kean-s": ["Jeff"]
}


if __name__ == "__main__":
    # gather_corpus("enron_mail", "enron.csv")
    # prepare_all_experiments_sets()
    prepare_experiment_set(5, 300, pd.read_csv("telegram.csv", index_col=0), "experiment_sets/telegram.csv")