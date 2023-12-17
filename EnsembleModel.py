
class EnsembleModel:
    def __init__(self, size_of_layer: int) -> None:
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_train_one_hot = None
        self.y_val = None
        self.y_val_one_hot = None
        self.size_of_layer = size_of_layer
        self.columns = None

        self.random_forest = None
        self.xgboost = None
        self.mlp = None
        self.encoder = None
        self.data_transformer = None
        self.scaler = None

    def fit_data(self, df: pd.DataFrame) -> None:
        df = df[["author", "text"]]
        self.df = df
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(df["text"], df["author"],
                                                                              test_size=0.2)

        # transforms labels to one-hot encoding
        self.encoder = LabelBinarizer()
        self.encoder.fit(self.y_train)
        self.y_train_one_hot = self.encoder.transform(self.y_train)
        self.y_train = np.argmax(self.y_train_one_hot, axis=1)
        self.y_val_one_hot = self.encoder.transform(self.y_val)
        self.y_val = np.argmax(self.y_val_one_hot, axis=1)

        # transforms text to count vector
        self.data_transformer = process_text.create_count_vector(self.x_train)
        self.x_train = process_text.transform_count_vector(self.x_train, self.data_transformer)
        self.columns = self.x_train.columns
        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_val = process_text.transform_count_vector(self.x_val, self.data_transformer)
        self.x_val = self.scaler.transform(self.x_val)

    def init_mlp(self):
        input_dim = self.x_train.shape[1]
        print("Input length: ", input_dim)
        dense_size = self.size_of_layer
        output_dim = self.encoder.classes_.shape[0]

        model = Sequential()
        model.add(Dense(dense_size, activation='relu', input_dim=input_dim, kernel_regularizer=regularizers.L2(l2=1e-2)))
        model.add(Dropout(0.5))
        model.add(Dense(dense_size, activation='relu', kernel_regularizer=regularizers.L2(l2=1e-2)))
        model.add(Dropout(0.5))
        model.add(Dense(dense_size, activation='relu', kernel_regularizer=regularizers.L2(l2=1e-2)))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.mlp = model

    def init_random_forest(self):
        self.random_forest = RandomForestClassifier(n_estimators=100, min_samples_split=2, bootstrap=True,
                                                    criterion="gini", min_samples_leaf=1)

    def init_xgboost(self):
        self.xgboost = XGBClassifier()

    def train_models(self):
        self.init_mlp()
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
        #self.mlp.fit(self.x_train, self.y_train_one_hot, epochs=100, validation_data=(self.x_val, self.y_val_one_hot),
        #             callbacks=[callback])

        self.init_random_forest()
        self.random_forest.fit(self.x_train, self.y_train)

        self.init_xgboost()
        self.xgboost.fit(self.x_train, self.y_train)

    def predict(self, df: pd.DataFrame):
        df = process_text.transform_tf_idf(df["text"], self.data_transformer)
        df = self.scaler.transform(df)

        # each model predicts the author
        random_forest_pred = self.random_forest.predict(df)
        random_forest_pred_onehot = OneHotEncoder(sparse=False).fit_transform(np.array(random_forest_pred.reshape(-1, 1)))
        xgboost_pred = self.xgboost.predict(df)
        xgboost_pred_onehot = OneHotEncoder(sparse=False).fit_transform(np.array(xgboost_pred.reshape(-1, 1)))
        mlp_pred_softmax = self.mlp.predict(df)
        mlp_pred_onehot = softmax_to_binary(mlp_pred_softmax)
        mlp_pred = np.argmax(mlp_pred_onehot, axis=1)

        # soft voting of all ensemble models
        pred_sum = np.sum([random_forest_pred_onehot, xgboost_pred_onehot, mlp_pred_onehot], axis=0)
        pred = np.argmax(pred_sum, axis=1)

        return pred, random_forest_pred, xgboost_pred, mlp_pred

    def rf_feature_importance(self):
        importances = self.random_forest.feature_importances_
        importances_labels = list(zip(self.columns, importances))
        df = pd.DataFrame(importances_labels, columns=["feature", "importance"])
        df = df.sort_values(by=["importance"], ascending=False)

        importances_xgb = self.xgboost.feature_importances_
        importances_xgb_labels = list(zip(self.columns, importances_xgb))
        df_xgb = pd.DataFrame(importances_xgb_labels, columns=["feature", "importance"])
        df_xgb = df_xgb.sort_values(by=["importance"], ascending=False)
        return importances

    def evaluate(self, df: pd.DataFrame):
        self.rf_feature_importance()

        y_test_one_hot = self.encoder.transform(df['author'])
        y_test = np.argmax(y_test_one_hot, axis=1)
        # x_test = process_text.transform_tf_idf(df["text"], self.data_transformer)

        predicted, rf_pred, xgb_pred, mlp_pred = self.predict(df)
        print("Ensemble accuracy: ", accuracy_score(y_true=y_test, y_pred=predicted))
        print("Random forest accuracy: ", accuracy_score(y_true=y_test, y_pred=rf_pred))
        print("XGB classifier accuracy: ", accuracy_score(y_true=y_test, y_pred=xgb_pred))
        print("MLP accuracy: ", accuracy_score(y_true=y_test, y_pred=mlp_pred))

        onehot_results = OneHotEncoder(sparse=False).fit_transform(np.array(predicted.reshape(-1, 1)))

        return self.encoder.inverse_transform(onehot_results)