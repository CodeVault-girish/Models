import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, concatenate, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer


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

def load_and_merge_features(file_1, file_2, label_file):
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)
    labels = pd.read_csv(label_file)
    
    labels = labels.rename(columns={'Participant_ID': 'Participant_ID'})
    for df in [df1, df2]:
        df['Participant_ID'] = df['filename'].str.split('_').str[0].astype(int)
    
    df1 = df1.merge(labels[['Participant_ID', 'PHQ_Score']], on='Participant_ID', how='left').drop(columns=['Participant_ID'])
    df2 = df2.merge(labels[['Participant_ID', 'PHQ_Score']], on='Participant_ID', how='left').drop(columns=['Participant_ID'])

    features_1 = df1.drop(columns=['PHQ_Score','filename']).values
    features_2 = df2.drop(columns=['PHQ_Score','filename']).values
    target = df1['PHQ_Score'].values
    
    return features_1, features_2, target

# def preprocess_features(features_1, features_2):
#     scaler = StandardScaler()
#     combined_features = np.concatenate([features_1, features_2], axis=0)
#     scaler.fit(combined_features)
#     features_1 = scaler.transform(features_1)
#     features_2 = scaler.transform(features_2)
#     return features_1[..., np.newaxis], features_2[..., np.newaxis]


def preprocess_features(features_1, features_2):
    scaler = StandardScaler()
    features_1 = scaler.fit_transform(features_1)
    features_2 = scaler.fit_transform(features_2)
    return features_1[..., np.newaxis], features_2[..., np.newaxis]  

def create_model(input_shape1, input_shape2):
    
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
    branch_1 = GlobalAveragePooling1D()(branch_1)
    
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
    branch_2 = GlobalAveragePooling1D()(branch_2)


    cka_similarity = CKALossLayer(name="cka_loss_layer")([branch_1, branch_2])
    
    merged = concatenate([branch_1, branch_2])
    merged = Dense(64, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    
    output = Dense(1, name="main_output")(merged)
    model = Model(inputs=[input_1, input_2], outputs=[output, cka_similarity])
    
    def custom_loss(y_true, y_pred):
        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        return mse_loss


    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'main_output': custom_loss, 'cka_loss_layer': lambda y_true, y_pred: tf.constant(0.0)},
        metrics={'main_output': [tf.keras.metrics.MeanAbsoluteError()]}
    )
    return model



def train_model(model, features_1_train, features_2_train, target_train, features_1_val, features_2_val, target_val, epochs=50):
    reduce_lr = ReduceLROnPlateau(monitor='val_main_output_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1, mode='min')
    early_stopping = EarlyStopping(monitor='val_main_output_loss', patience=10, restore_best_weights=True, mode='min')

    model.fit(
        [features_1_train, features_2_train],
        [target_train, np.zeros(len(target_train))],  
        validation_data=([features_1_val, features_2_val], [target_val, np.zeros(len(target_val))]),
        epochs=epochs,
        batch_size=32,
        callbacks=[reduce_lr, early_stopping]
    )



def evaluate_model(model, features_1_test, features_2_test, target_test):
    predictions = model.predict([features_1_test, features_2_test])[0] 
    mae = mean_absolute_error(target_test, predictions)
    rmse = np.sqrt(mean_squared_error(target_test, predictions))
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    
def main(file_1_train, file_1_test, file_1_val, file_2_train, file_2_test, file_2_val, label_file, epochs=50):
    features_1_train, features_2_train, target_train = load_and_merge_features(file_1_train, file_2_train, label_file)
    features_1_test, features_2_test, target_test = load_and_merge_features(file_1_test, file_2_test, label_file)
    features_1_val, features_2_val, target_val = load_and_merge_features(file_1_val, file_2_val, label_file)

    features_1_train, features_2_train = preprocess_features(features_1_train, features_2_train)
    features_1_test, features_2_test = preprocess_features(features_1_test, features_2_test)
    features_1_val, features_2_val = preprocess_features(features_1_val, features_2_val)
    
    input_shape1 = (features_1_train.shape[1],1)
    input_shape2 = (features_2_train.shape[1],1)  
    print(input_shape1)
    print(input_shape2)

    model = create_model(input_shape1, input_shape2)
    train_model(model, features_1_train, features_2_train, target_train, features_1_val, features_2_val, target_val, epochs=epochs)  

    evaluate_model(model, features_1_test, features_2_test, target_test)





# if __name__ == "__main__":


    # test_labels = pd.read_csv('/kaggle/input/labels/EDAIC_LABELS/test_split.csv')
    # train_labels = pd.read_csv('/kaggle/input/labels/EDAIC_LABELS/train_split.csv')
    # val_labels = pd.read_csv('/kaggle/input/labels/EDAIC_LABELS/dev_split.csv')
    # all_labels = pd.concat([test_labels, train_labels, val_labels])

    # label_file = '/kaggle/working/all_labels.csv'
    # all_labels.to_csv(label_file, index=False)

    # file_1_train = '/kaggle/input/depression/EDAIC/EDAIC/MFCC_train.csv'
    # file_1_test = '/kaggle/input/depression/EDAIC/EDAIC/MFCC_test.csv'
    # file_1_val = '/kaggle/input/depression/EDAIC/EDAIC/MFCC_val.csv'

    # file_2_train = '/kaggle/input/depression/EDAIC/EDAIC/UniSpeechSAT_train.csv'
    # file_2_test = '/kaggle/input/depression/EDAIC/EDAIC/UniSpeechSAT_test.csv'
    # file_2_val = '/kaggle/input/depression/EDAIC/EDAIC/UniSpeechSAT_val.csv'

    # epochs = 50

    # main(file_1_train, file_1_test, file_1_val, file_2_train, file_2_test, file_2_val, label_file, epochs=epochs)
