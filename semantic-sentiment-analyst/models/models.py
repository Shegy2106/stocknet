from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Bidirectional, Lambda, TimeDistributed, Permute, Flatten, Multiply, Activation, RepeatVector, GlobalAveragePooling1D
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow import reduce_sum, expand_dims
import tensorflow


def simpleLSTM(time_step, features, optimizer, loss):
    model = Sequential()

    model.add(InputLayer(shape=(time_step, features)))

    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(units=1))

    model.compile(optimizer=optimizer, loss=loss)

    return model


def stackedLSTM(time_step, optimizer, loss):
    model = Sequential()

    model.add(InputLayer(shape=(time_step, 1)))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Second LSTM Layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Third LSTM Layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(units=1))

    model.compile(optimizer=optimizer, loss=loss)

    return model


def bidirectionalLSTM(time_step, optimizer, loss):
    model = Sequential()

    model.add(InputLayer(shape=(time_step, 1)))

    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))

    # Second LSTM Layer
    model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(units=1))

    model.compile(optimizer=optimizer, loss=loss)

    return model


def LSTMAttentionMechanism(time_step, optimizer, loss):
    model = Sequential()

    model.add(InputLayer(shape=(time_step, 1)))

    model.add(LSTM(units=50, return_sequences=True))

    model.add(Attention())

    model.add(Dense(units=1))

    model.compile(optimizer=optimizer, loss=loss)

    return model


class Attention(tensorflow.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        self.dense = Dense(1, activation='tanh')  # Define the dense layer in __init__
        self.flatten = Flatten()
        self.activation = Activation('softmax')
        self.repeat_vector = None
        self.permute = None

    def build(self, input_shape):
        # Initialize repeat vector and permute layers based on input shape
        self.repeat_vector = RepeatVector(input_shape[-1])
        self.permute = Permute([2, 1])
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        # Compute the attention scores
        attention_scores = self.dense(inputs)
        attention_scores = self.flatten(attention_scores)
        attention_weights = self.activation(attention_scores)
        attention_weights = self.repeat_vector(attention_weights)
        attention_weights = self.permute(attention_weights)

        # Apply the attention weights to the inputs
        weighted_inputs = Multiply()([inputs, attention_weights])
        output = reduce_sum(weighted_inputs, axis=1)
        return output


def encoderDecoderLSTM(time_step, optimizer, loss):
    model = Sequential()

    model.add(InputLayer(shape=(time_step, 1)))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling1D())
    model.add(RepeatVector(time_step))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(TimeDistributed(Dense(units=1)))

    model.compile(optimizer=optimizer, loss=loss)

    return model
