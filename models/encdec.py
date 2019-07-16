import tensorflow as tf
import numpy as np
from .nmtbase import NMTBase


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, units, batch_size, num_layers, dropout):
        super(Encoder, self).__init__()

        self.units = units
        self.batch_size = batch_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.num_layers = num_layers

        self.lstms = [tf.keras.layers.LSTM(units, 
                        return_state = True,
                        return_sequences = True,
                        recurrent_initializer='glorot_uniform',
                        dropout = dropout)
                        for i in range(num_layers)]

    def call(self, x, hidden = None, training=False):

        batch_size = x.shape[0]

        if hidden is None:
            hidden = self.initialize_hidden_state(batch_size)

        input_mask = self.embedding.compute_mask(x)
        x = self.embedding(x)

        states = []
        for i, lstm in enumerate(self.lstms):

            x, h, c = lstm(x, initial_state = hidden[i], training=training, mask=input_mask)

            states.append((h, c))

        return states


    def initialize_hidden_state(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        return [(tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))) for i in range(self.num_layers)]


    def extract_end_states(self, data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        return tf.gather_nd(data, indices)


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size, num_layers, dropout):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.units = units
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.num_layers = num_layers

        self.lstms = [tf.keras.layers.LSTM(units,
                        return_sequences = True,
                        return_state = True,
                        recurrent_initializer = 'glorot_uniform',
                        dropout = dropout)
                        for i in range(num_layers)]

        self.logits = tf.keras.layers.Dense(vocab_size)


    def call(self, x, hidden, training=False):

        input_mask = self.embedding.compute_mask(x)
        x = self.embedding(x)

        batch_size = x.shape[0]

        states = []
        for i, lstm in enumerate(self.lstms):
            x, h, c = lstm(x, initial_state = hidden[i], training=training, mask=input_mask)
            states.append((h, c))

        outputs = self.logits(x)
        outputs = tf.reshape(outputs, [batch_size, -1, self.vocab_size])

        return outputs, states

    def initialize_hidden_state(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        return [(tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))) for i in range(self.num_layers)]



class EncDec(NMTBase):

    def __init__(self, **kwargs):

        super(EncDec, self).__init__(**kwargs)

        self.encoder = Encoder(self.vocab1_size, self.embedding_size, self.units, self.batch_size, self.num_layers, self.dropout)
        self.decoder = Decoder(self.vocab2_size, self.embedding_size, self.units, self.batch_size, self.num_layers, self.dropout)


    def forward(self, X, dec_X, training=False):

        enc_hidden = self.encoder(X, training=training)

        logits, dec_hidden = self.decoder(dec_X, hidden = enc_hidden, training=training)

        return logits


    def train_step(self, X, X_lengths, Y, Y_lengths, ids):

        dec_X = Y[:, :-1]
        dec_Y = Y[:, 1:]

        logits = self.forward(X, dec_X, training=True)

        return self.loss(logits=logits, targets=dec_Y, target_lengths = Y_lengths - 1)


    def encode(self, X):
        return self.encoder(X)


    def decode(self, dec_X, enc_hidden):
        return self.decoder(dec_X, hidden=enc_hidden)


    def get_variables(self):
        return self.encoder.variables + self.decoder.variables

    def get_checkp(self):

        return dict(encoder = self.encoder, decoder = self.decoder)
