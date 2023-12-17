class LstmModel:
    def __init__(self, df: pd.DataFrame, embed_letters: bool = False, limited_len: bool = True, embed_dim: int = 256,
                 batch_ratio: float = 1, max_len: int = 256) -> None:
        self.df = df
        self.embed_letters = embed_letters
        self.limited_len = limited_len
        self.embed_dim = embed_dim  # size of vector to which words/letters are embedded
        self.batch_ratio = batch_ratio
        self.max_len = max_len
        self.encoder = None
        self.tok = None
        self.model = None

    # builds neural network architecture for lstm model
    def build_network(self, max_len: int, vocab_size: int, embed_matrix, encoder) -> Model:
        input1 = Input(shape=(max_len,))
        embed = Embedding(input_dim=vocab_size, output_dim=self.embed_dim, input_length=max_len,
                          embeddings_initializer=Constant(embed_matrix))(input1)
        lstm = Bidirectional(LSTM(256, return_sequences=True))(embed)  # jde zkusit bez return sequences
        maxpool = GlobalMaxPooling1D()(lstm)
        drop = Dropout(0.5)(maxpool)
        softmax = Dense(encoder.classes_.shape[0], activation="softmax")(drop)

        dense = Dense(256)(softmax)
        dropout = Dropout(0.5)(dense)
        output = Dense(encoder.classes_.shape[0], activation='softmax')(dropout)
        model = Model(inputs=[input1], outputs=output)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                      metrics=['categorical_accuracy'])

        return model

    def calculate_max_len(self, x_texts: pd.Series) -> int:
        if self.limited_len:
            max_len = self.max_len
        else:
            if self.embed_letters:
                x_texts_list = x_texts.tolist()
                x_texts_len = list(map(lambda x: len(x), x_texts_list))
                max_len = max(x_texts_len)
            else:
                x_texts_list = x_texts.tolist()
                x_texts_len = list(map(lambda x: len(x.split()), x_texts_list))
                max_len = max(x_texts_len)

        return max_len

    def slice_batch(self, df_to_slice: pd.DataFrame, iter_i: int) -> pd.DataFrame:
        lower_index = floor(iter_i*self.batch_ratio*len(df_to_slice))
        upper_index = floor((iter_i+1)*self.batch_ratio*len(df_to_slice))
        return df_to_slice[lower_index:upper_index]

    def build_embedding_matrix(self, tok: Tokenizer, word_vec_dict: dict, vocab_size: int) -> np.ndarray:
        embed_matrix = np.zeros(shape=(vocab_size, self.embed_dim))
        for word, i in tok.word_index.items():
            embed_vector = word_vec_dict.get(word)
            if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
                embed_matrix[i] = embed_vector

        return embed_matrix

    def get_corpus_and_df_text(self, x_texts: pd.DataFrame) -> tuple:
        if self.embed_letters:
            text_list = x_texts.tolist()
            text_list = [text.replace(" ", "") for text in text_list]
            corpus_text = [[*text] for text in text_list]

            df_text = self.df["text"].tolist()
            df_text = [text.replace(" ", "") for text in df_text]
            df_text = [[*text] for text in df_text]
        else:
            corpus_text = x_texts
            df_text = self.df['text']

        return corpus_text, df_text

    def build_word_vec_dict(self, x_train: pd.Series) -> dict:
        if self.embed_letters:
            w2v_model = process_text.create_word2vec_letters(x_train)
        else:
            w2v_model = process_text.create_word2vec(x_train)

        vocab = w2v_model.wv.key_to_index
        vocab = list(vocab.keys())
        word_vec_dict = {}
        for word in vocab:
            word_vec_dict[word] = w2v_model.wv.get_vector(word)

        return word_vec_dict

    # lstm model with word2vec embedding, option to use words or letters and length of input text
    def run_lstm_model(self) -> None:
        x_train, x_val, y_train, y_val = train_test_split(self.df["text"], self.df['author'], test_size=0.2)

        self.encoder = LabelBinarizer()
        y_train = self.encoder.fit_transform(y_train)
        y_val = self.encoder.transform(y_val)

        # translation dictionary from word/letter to vector
        word_vec_dict = self.build_word_vec_dict(x_train)

        # corpus text is used for training the tokenizer, df_text is used for tokenization of the whole dataset
        corpus_text, df_text = self.get_corpus_and_df_text(x_train)

        self.tok = Tokenizer()
        self.tok.fit_on_texts(corpus_text)
        tokenized_text = self.tok.texts_to_sequences(df_text)

        self.max_len = self.calculate_max_len(x_train)
        pad_rev = pad_sequences(tokenized_text, maxlen=self.max_len, padding='post')

        x_train_ind = x_train.index
        x_train_input1 = pad_rev[x_train_ind]
        x_val_input1 = pad_rev[x_val.index]

        vocab_size = len(self.tok.word_index) + 1
        embed_matrix = self.build_embedding_matrix(self.tok, word_vec_dict, vocab_size)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.model = self.build_network(max_len=self.max_len, vocab_size=vocab_size, embed_matrix=embed_matrix,
                                        encoder=self.encoder)

        self.model.fit([x_train_input1], y_train, epochs=1000, validation_data=([x_val_input1], y_val),
                       batch_size=64, callbacks=[callback])

    def evaluate(self, df: pd.Series):
        y_test = self.encoder.transform(df['author'])
        y_test = np.argmax(y_test, axis=1)

        text_list = df["text"].tolist()
        text_list = [text.replace(" ", "") for text in text_list]
        corpus_text = [[*text] for text in text_list]
        tokenized_text = self.tok.texts_to_sequences(corpus_text)
        pad_rev = pad_sequences(tokenized_text, maxlen=self.max_len, padding='post')

        predicted_onehot = self.model.predict(pad_rev)
        predicted = np.argmax(predicted_onehot, axis=1)
        print("BiLSTM accuracy: ", accuracy_score(y_true=y_test, y_pred=predicted))

        predicted_names = self.encoder.inverse_transform(predicted_onehot)
        return predicted_names
