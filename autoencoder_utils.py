import tensorflow as tf

import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, Masking

@keras.saving.register_keras_serializable(package="Project")
class lstm_bottleneck(tf.keras.layers.Layer):
    def __init__(self, units, time_steps, **kwargs):
        self.units = units
        self.time_steps = time_steps
        self.lstm_layer = LSTM(units, return_sequences=False)
        self.repeat_layer = RepeatVector(time_steps)
        super(lstm_bottleneck, self).__init__(**kwargs)
    
    def call(self, inputs):
        return self.repeat_layer(self.lstm_layer(inputs))
    
    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape):
        pass

def autoencoder_model(Tx, n_a, n_values):
    input_layer = Input(shape=(Tx, n_values), name='input') 
    X = Masking(mask_value=0, name='masking')(input_layer)
    enc_LSTM = lstm_bottleneck(units=n_a, time_steps=Tx, name='encoder_cell')
    encoded = enc_LSTM(X)
    encoder = Model(inputs=X, outputs=encoded, name='encoder')
    
    dec_LSTM = LSTM(units=n_a, return_sequences=True, name='decoder_cell')
    decoded = dec_LSTM(encoded)

    outputs = Dense(n_values, activation=None, name='output')(decoded)
    decoder = Model(inputs=encoded, outputs=outputs, name='decoder')
    autoencoder = Model(inputs=input_layer, outputs=outputs, name='autoencoder')
    return autoencoder, encoder, decoder

def extract_encoder(autoencoder):
    input_layer = autoencoder.get_layer('input')
    batch_shape = input_layer.get_config()['batch_shape']
    input = Input(shape=(batch_shape[1], batch_shape[2]), name='input') 
    masking_layer = autoencoder.get_layer('masking')
    encoder_cell = autoencoder.get_layer('encoder_cell').lstm_layer
    encoder = Model(inputs=input, outputs=encoder_cell(masking_layer(input)))
    return encoder