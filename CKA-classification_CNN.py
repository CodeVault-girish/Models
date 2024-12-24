class CKALossLayer(Layer):
    def __init__(self, **kwargs):
        super(CKALossLayer, self).__init__(**kwargs)

    def gram_matrix(self, X):
        return tf.linalg.matmul(X, X, transpose_b=True)

    def center_gram_matrix(self, K):
        n = tf.shape(K)[0]
        one_n = tf.ones((n, n), dtype=tf.float32) / tf.cast(n, tf.float32)
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    def hsic(self, K, L):
        return tf.linalg.trace(tf.linalg.matmul(K, L))

    def call(self, inputs):
        X, Y = inputs
        K = self.gram_matrix(X)
        L = self.gram_matrix(Y)
        Kc = self.center_gram_matrix(K)
        Lc = self.center_gram_matrix(L)
        hsic_xy = self.hsic(Kc, Lc)
        hsic_xx = self.hsic(Kc, Kc)
        hsic_yy = self.hsic(Lc, Lc)
        cka_similarity = -hsic_xy / tf.sqrt(hsic_xx * hsic_yy)
        return cka_similarity

def self_attention(inputs, num_heads=4):
    return MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1] if inputs.shape[-1] is not None else 64)(inputs, inputs)


def load_and_merge_features(file_path_hc_1, file_path_pt_1, file_path_hc_2, file_path_pt_2):
    df_hc_1 = pd.read_csv(file_path_hc_1)
    df_pt_1 = pd.read_csv(file_path_pt_1)
    df_hc_2 = pd.read_csv(file_path_hc_2)
    df_pt_2 = pd.read_csv(file_path_pt_2)

    df_hc_1['class'] = 'hc'
    df_pt_1['class'] = 'pt'
    df_hc_2['class'] = 'hc'
    df_pt_2['class'] = 'pt'
    
    df_1 = pd.concat([df_hc_1, df_pt_1], ignore_index=True)
    df_2 = pd.concat([df_hc_2, df_pt_2], ignore_index=True)
    df_merged = pd.merge(df_1, df_2, on=['filename', 'class'], suffixes=('_set1', '_set2'))
    
    feature_columns_1 = df_merged.filter(regex='_set1$').columns
    features_1 = df_merged[feature_columns_1].values
    feature_columns_2 = df_merged.filter(regex='_set2$').columns
    features_2 = df_merged[feature_columns_2].values
    
    labels = df_merged['class'].values
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded)

    print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print("Encoded Labels Sample:", labels_encoded[:5])
    # Add this to check class distribution after merging
    print("Class Distribution:", df_merged['class'].value_counts())

    return features_1, features_2, labels_encoded


def preprocess_features(features_1, features_2):
    scaler = StandardScaler()
    features_1 = scaler.fit_transform(features_1)
    features_2 = scaler.fit_transform(features_2)
    return features_1[..., np.newaxis], features_2[..., np.newaxis]

def create_classification_model(input_shape1, input_shape2):
    input_1 = Input(shape=input_shape1)
    branch_1 = Conv1D(64, kernel_size=3, activation='relu')(input_1)
    branch_1 = BatchNormalization()(branch_1)
    branch_1 = MaxPooling1D(pool_size=2)(branch_1)
    branch_1 = Dropout(0.3)(branch_1)
    branch_1 = Conv1D(128, kernel_size=3, activation='relu')(branch_1)
    branch_1 = BatchNormalization()(branch_1)
    branch_1 = MaxPooling1D(pool_size=2)(branch_1)
    branch_1 = Dropout(0.3)(branch_1)
    branch_1 = self_attention(branch_1)
    branch_1 = Flatten()(branch_1)

    
    input_2 = Input(shape=input_shape2)
    branch_2 = Conv1D(64, kernel_size=3, activation='relu')(input_2)
    branch_2 = BatchNormalization()(branch_2)
    branch_2 = MaxPooling1D(pool_size=2)(branch_2)
    branch_2 = Dropout(0.3)(branch_2)
    branch_2 = Conv1D(128, kernel_size=3, activation='relu')(branch_2)
    branch_2 = BatchNormalization()(branch_2)
    branch_2 = MaxPooling1D(pool_size=2)(branch_2)
    branch_2 = Dropout(0.3)(branch_2)
    branch_2 = self_attention(branch_2)
    branch_2 = Flatten()(branch_2)

    cka_similarity = CKALossLayer(name="cka_loss_layer")([branch_1, branch_2])
    
    merged = concatenate([branch_1, branch_2])
    merged = Dense(64, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    
    output = Dense(2, activation='softmax', name="main_output")(merged)
    model = Model(inputs=[input_1, input_2], outputs=[output, cka_similarity])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'main_output': 'categorical_crossentropy', 'cka_loss_layer': lambda y_true, y_pred: tf.constant(0.0)},
        metrics={'main_output': 'accuracy'}
    )
    return model

def train_model(model, features_1_train, features_2_train, target_train, features_1_test, features_2_test, target_test, epochs=50):
    reduce_lr = ReduceLROnPlateau(monitor='val_main_output_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1, mode='min')
    early_stopping = EarlyStopping(monitor='val_main_output_loss', patience=10, restore_best_weights=True, mode='min')

    model.fit(
        [features_1_train, features_2_train],
        [target_train, np.zeros(len(target_train))],
        validation_data=([features_1_test, features_2_test], [target_test, np.zeros(len(target_test))]),
        epochs=epochs,
        batch_size=32,
        callbacks=[reduce_lr, early_stopping]
    )


def evaluate_model(model, features_1_test, features_2_test, target_test):
    predictions = model.predict([features_1_test, features_2_test])[0]
    predictions = np.argmax(predictions, axis=1)
    target_test = np.argmax(target_test, axis=1)

    print("Predictions Sample:", predictions[:5])
    print("Target Sample:", target_test[:5])

    accuracy = accuracy_score(target_test, predictions)
    report = classification_report(target_test, predictions, digits=4)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)


def main(file_path_hc_1, file_path_pt_1, file_path_hc_2, file_path_pt_2, epochs=50):
    features_1, features_2, target = load_and_merge_features(file_path_hc_1, file_path_pt_1, file_path_hc_2, file_path_pt_2)

    features_1, features_2 = preprocess_features(features_1, features_2)

    split = int(0.8 * len(features_1))
    features_1_train, features_1_test = features_1[:split], features_1[split:]
    features_2_train, features_2_test = features_2[:split], features_2[split:]
    target_train, target_test = target[:split], target[split:]
    
    input_shape1 = (features_1_train.shape[1], 1)
    input_shape2 = (features_2_train.shape[1], 1)

    model = create_classification_model(input_shape1, input_shape2)
    train_model(model, features_1_train, features_2_train, target_train, features_1_test, features_2_test, target_test, epochs=epochs)
    

    features_1_train, features_1_test, features_2_train, features_2_test, target_train, target_test = train_test_split(features_1, features_2, target, test_size=0.2, stratify=target, random_state=42)

    evaluate_model(model, features_1_test, features_2_test, target_test)




'''
use the own load_and_merge_features accordig to the your data have to change the preprocess_features or main function
'''


# if __name__ == "__main__":
    # file_path_hc_1 = '/kaggle/input/depression/ANDROID/ANDROID/SNAC_interview_hc.csv'
    # file_path_pt_1 = '/kaggle/input/depression/ANDROID/ANDROID/SNAC_interview_pt.csv'
    # file_path_hc_2 = '/kaggle/input/depression/ANDROID/ANDROID/HuBERT_interview_hc.csv'
    # file_path_pt_2 = '/kaggle/input/depression/ANDROID/ANDROID/HuBERT_interview_pt.csv'

    # main(file_path_hc_1, file_path_pt_1, file_path_hc_2, file_path_pt_2, epochs=50)