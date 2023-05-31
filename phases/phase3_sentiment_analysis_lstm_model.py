import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Read in the preprocessed dataset
dataset = pd.read_csv('../csv_files/processed_data/preprocessed_data.tsv', sep='\t')
dataset['avg_vector'] = dataset['avg_vector'].apply(lambda x: ast.literal_eval(x))

# Create features and labels
features = np.vstack(dataset['avg_vector'].values)
labels = dataset['cluster']

# Separate into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Change labels to categorical representation
labels_train_cat = to_categorical(labels_train)
labels_test_cat = to_categorical(labels_test)

# Reshape the features to meet LSTM model requirements
features_train = np.reshape(features_train, (features_train.shape[0], features_train.shape[1], 1))
features_test = np.reshape(features_test, (features_test.shape[0], features_test.shape[1], 1))

# Construct LSTM model structure
lstm_model = Sequential()
lstm_model.add(LSTM(128, input_shape=(features_train.shape[1], features_train.shape[2])))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(labels_train_cat.shape[1], activation='softmax'))

# Set up model for training
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Execute model training
model_history = lstm_model.fit(features_train, labels_train_cat, epochs=100, batch_size=128, validation_data=(features_test, labels_test_cat))

# Assess model performance on test data
evaluation_loss, evaluation_accuracy = lstm_model.evaluate(features_test, labels_test_cat)
print(f'Test Loss: {evaluation_loss:.3f}\nTest Accuracy: {evaluation_accuracy:.3f}')
