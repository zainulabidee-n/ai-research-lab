import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    SimpleRNN,
    LSTM,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam()

# Load dataset
_df = pd.read_csv("BTC-2017min.csv")

print(_df.head())
print(_df.shape)
print(_df.isnull().sum())

# plt.figure(figsize=(12,6))
# plt.plot(_df['close'])
# plt.title("Bitcoin Closing Price")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.show()

_df = _df.dropna()

scaler = MinMaxScaler()

close_prices = _df[['close']]
scaled_data = scaler.fit_transform(close_prices)


sequence_length = 30


def create_sequences(data, seq_length):
    X = []
    y = []

    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])

    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data, sequence_length)

print(X.shape)
print(y.shape)


split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

print(X_train.shape)
print(X_test.shape)


#Simple RNN Model

# rnn_model = Sequential([
#     SimpleRNN(32, return_sequences=False, input_shape=(30,1)),
#     Dropout(0.2),
#     Dense(1)
# ])

# rnn_model.compile(
#     optimizer=optimizer,
#     loss='mse'
# )

# rnn_history = rnn_model.fit(
#     X_train,
#     y_train,
#     epochs=10,
#     batch_size=32,
#     validation_data=(X_test, y_test)
# )

# plt.plot(rnn_history.history['loss'], label='Training Loss')
# plt.plot(rnn_history.history['val_loss'], label='Validation Loss')

# plt.title('RNN Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#LSTM Model

# lstm_model = Sequential([
#     LSTM(64, input_shape=(30,1)),
#     Dropout(0.2),
#     Dense(1)
# ])

# lstm_model.compile(
#     optimizer=optimizer,
#     loss='mse'
# )

# lstm_history = lstm_model.fit(
#     X_train,
#     y_train,
#     epochs=10,
#     batch_size=32,
#     validation_data=(X_test, y_test)
# )

# plt.plot(lstm_history.history['loss'], label='Training Loss')
# plt.plot(lstm_history.history['val_loss'], label='Validation Loss')

# plt.title('LSTM Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

import numpy as np
import tensorflow as tf

# Positional Encoding Function
def positional_encoding(sequence_length, embedding_dim):

    pos_encoding = np.zeros((sequence_length, embedding_dim))

    for pos in range(sequence_length):

        for i in range(0, embedding_dim, 2):

            angle = pos / np.power(10000, (2 * i) / embedding_dim)

            pos_encoding[pos, i] = np.sin(angle)

            if i + 1 < embedding_dim:
                pos_encoding[pos, i + 1] = np.cos(angle)

    return tf.cast(pos_encoding, dtype=tf.float32)


# Example
pos_encoding = positional_encoding(30, 32)

print(pos_encoding.shape)

sequence_len = 30
embed_dim = 32
num_heads = 4

inputs = Input(shape=(30,1))

x = Dense(embed_dim)(inputs)

pos_encoding = positional_encoding(sequence_len, embed_dim)
x = x + pos_encoding

attention_output = MultiHeadAttention(
    num_heads=num_heads,
    key_dim=embed_dim
)(x, x)

x = LayerNormalization(epsilon=1e-6)(x + attention_output)

ffn = Dense(64, activation='relu')(x)
ffn = Dense(embed_dim)(ffn)

x = LayerNormalization(epsilon=1e-6)(x + ffn)

x = GlobalAveragePooling1D()(x)

outputs = Dense(1)(x)

transformer_model = Model(inputs, outputs)
transformer_model.compile(
    optimizer=optimizer,
    loss='mse'
)

transformer_history = transformer_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)