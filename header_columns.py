stylometry_names = ["num_of_words", "num_of_sentences", "num_of_lines", "num_of_uppercase", "num_of_titlecase",
                    "average_len_of_words", "num_of_punctuation", "num_of_special_chars", "num_of_chars",
                    "num_of_stopwords", "num_of_unique_words", "num_of_digits"]

header_metadata_columns = ["sent_hour", "subject_num_of_words", "subject_num_of_char", "subject_num_of_uppercase_char",
                           "num_od_numeric_char", "num_of_punctuation_marks", "num_of_addressees",
                           "num_of_addressees_from_same_domain", "num_of_cc", "num_of_cc_from_same_domain"]
all_stylometry = header_metadata_columns + stylometry_names