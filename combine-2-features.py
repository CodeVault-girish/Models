import os
import re
import gc
from collections import Counter

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
    BatchNormalization, concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam





# Set GPU memory
def set_gpu_memory(gpu_id=0):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"Using GPU {gpu_id}")
        except RuntimeError as e:
            print(f"Could not set GPU {gpu_id}: {e}")

# Load and preprocess data
def load_and_preprocess_chunked(file_path1, file_path2, prefix1='prefix1_', prefix2='prefix2_'):
    data1 = pd.read_csv(file_path1, dtype={col: np.float32 for col in pd.read_csv(file_path1, nrows=1).select_dtypes(include=[np.number]).columns})
    data2 = pd.read_csv(file_path2, dtype={col: np.float32 for col in pd.read_csv(file_path2, nrows=1).select_dtypes(include=[np.number]).columns})
    
    label = int(re.findall(r'\d+', file_path1.split('/')[-1])[0]) - 1  # Adjust if label extraction changes
    data1 = data1.add_prefix(prefix1)
    data2 = data2.add_prefix(prefix2)
    
    aligned_data = pd.merge(data1, data2, left_on=f'{prefix1}name', right_on=f'{prefix2}name', how='inner').drop([f'{prefix1}name', f'{prefix2}name'], axis=1)
    labels = np.full((aligned_data.shape[0],), label, dtype=np.int32)
    return aligned_data, labels

# Build CNN model
def build_model(input_shape1, input_shape2, num_classes):
    input1 = Input(shape=input_shape1)
    branch1 = Conv1D(32, kernel_size=3, activation='relu')(input1)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling1D(pool_size=2)(branch1)
    branch1 = Dropout(0.2)(branch1)
    branch1 = Conv1D(64, kernel_size=3, activation='relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling1D(pool_size=2)(branch1)
    branch1 = Dropout(0.2)(branch1)
    branch1 = Flatten()(branch1)

    input2 = Input(shape=input_shape2)
    branch2 = Conv1D(32, kernel_size=3, activation='relu')(input2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling1D(pool_size=2)(branch2)
    branch2 = Dropout(0.2)(branch2)
    branch2 = Conv1D(64, kernel_size=3, activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling1D(pool_size=2)(branch2)
    branch2 = Dropout(0.2)(branch2)
    branch2 = Flatten()(branch2)

    concatenated = concatenate([branch1, branch2])
    x = Dense(64, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Compute EER
def compute_eer(label, pred, positive_label=1):
    fpr, tpr, thresholds = roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = (fpr[np.nanargmin(np.abs(fnr - fpr))] + fnr[np.nanargmin(np.abs(fnr - fpr))]) / 2
    return eer

def compute_multiclass_eer(y_true, y_scores):
    n_classes = y_scores.shape[1]
    eers = []
    for i in range(n_classes):
        binary_labels = (y_true == i).astype(int)
        class_scores = y_scores[:, i]
        eer = compute_eer(binary_labels, class_scores)
        eers.append(eer)
        print(f"EER for class {i}: {eer}")
    average_eer = np.mean(eers)
    print(f"Average Equal Error Rate (EER): {average_eer}")
    return eers, average_eer

# Main function
def main(filenames, train_dir1, test_dir1, train_dir2, test_dir2, prefix1='prefix1_', prefix2='prefix2_'):
    set_gpu_memory(0)
    train_features, train_labels = [], []
    test_features, test_labels = [], []

    for filename in filenames:
        train_data, train_label = load_and_preprocess_chunked(
            os.path.join(train_dir1, filename),
            os.path.join(train_dir2, filename),
            prefix1, prefix2
        )
        train_features.append(train_data)
        train_labels.append(train_label)

        test_data, test_label = load_and_preprocess_chunked(
            os.path.join(test_dir1, filename),
            os.path.join(test_dir2, filename),
            prefix1, prefix2
        )
        test_features.append(test_data)
        test_labels.append(test_label)

        gc.collect()

    train_features = pd.concat(train_features, ignore_index=True)
    train_labels = np.concatenate(train_labels)
    test_features = pd.concat(test_features, ignore_index=True)
    test_labels = np.concatenate(test_labels)

    num_classes = len(np.unique(train_labels))
    train_labels_categorical = to_categorical(train_labels, num_classes)
    test_labels_categorical = to_categorical(test_labels, num_classes)

    train_X1, val_X1, train_X2, val_X2, train_y, val_y = train_test_split(
        train_features.filter(regex=f'^{prefix1}').values.reshape(-1, train_features.filter(regex=f'^{prefix1}').shape[1], 1),
        train_features.filter(regex=f'^{prefix2}').values.reshape(-1, train_features.filter(regex=f'^{prefix2}').shape[1], 1),
        train_labels_categorical, test_size=0.2, random_state=42
    )

    input_shape1 = (train_X1.shape[1], 1)
    input_shape2 = (train_X2.shape[1], 1)

    model = build_model(input_shape1, input_shape2, num_classes)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

    try:
        history = model.fit([train_X1, train_X2], train_y, epochs=20, batch_size=16,
                            validation_data=([val_X1, val_X2], val_y), callbacks=[early_stopping, reduce_lr])
    except tf.errors.ResourceExhaustedError as e:
        print(f"Memory issue encountered: {e}. Switching to GPU 1.")
        set_gpu_memory(1)
        history = model.fit([train_X1, train_X2], train_y, epochs=50, batch_size=16,
                            validation_data=([val_X1, val_X2], val_y), callbacks=[early_stopping, reduce_lr])

    test_X1 = test_features.filter(regex=f'^{prefix1}').values.reshape(-1, test_features.filter(regex=f'^{prefix1}').shape[1], 1)
    test_X2 = test_features.filter(regex=f'^{prefix2}').values.reshape(-1, test_features.filter(regex=f'^{prefix2}').shape[1], 1)
    test_loss, test_accuracy = model.evaluate([test_X1, test_X2], test_labels_categorical)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    test_predictions = model.predict([test_X1, test_X2])
    test_true_labels = np.argmax(test_labels_categorical, axis=1)
    test_predictions_labels = np.argmax(test_predictions, axis=1)
    print(classification_report(test_true_labels, test_predictions_labels, digits=4))

    eers, average_eer = compute_multiclass_eer(test_true_labels, test_predictions)
    print(f"Final Average EER: {average_eer * 100:.2f}%")




    
# filenames = ["A01_concatenated.csv", 
#              "A02_concatenated.csv", 
#              "A03_concatenated.csv",
#              "A04_concatenated.csv",
#              "A05_concatenated.csv",
#              "A06_concatenated.csv",
#              "A07_concatenated.csv",
#              "A08_concatenated.csv"
#             ]
#if __name__ == "__main__":
    # main(filenames, m_a_pMERT_v1_330M_MERT_train_deepfake, m_a_pMERT_v1_330M_MERT_dev_deepfake, I_B_train, I_B_dev, prefix1='m_', prefix2='I_')
