import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np 


class RNNPB(tf.keras.Model):
    
    _hard_pbs = None
    
    def __init__(self, vocab_size, embedding_size, units, num_layers, num_PB, num_sequences, pb_lr, gradient_clip, batch_size, sigma, dropout, bind_hard = False):
        super(RNNPB, self).__init__()
        
        self.units = units
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_PB = num_PB
        self.num_sequences = num_sequences
        self.pb_lr = pb_lr
        self.gradient_clip = gradient_clip
        self.sigma = sigma
        self.dropout = dropout
        
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)
        self.pb_embedding = self._get_pb_layer(bind_hard)
        
        self.lstms = [tf.keras.layers.LSTM(units,
                         return_state = True,
                         return_sequences = True,
                         recurrent_initializer='glorot_uniform',
                         dropout = self.dropout)
                         for i in range(num_layers)]
        
        self.logits = tf.keras.layers.Dense(vocab_size)

    
    def _get_pb_layer(self, bind_hard):
        
        def pb_layer():
            return tf.keras.layers.Embedding(self.num_sequences, self.num_PB, embeddings_initializer = 'zeros')
        
        if bind_hard:
            if RNNPB._hard_pbs is None:
                RNNPB._hard_pbs = pb_layer()
            
            return RNNPB._hard_pbs
        else:
            return pb_layer()

        
    def call(self, x, ids = None, pbs = None, hidden = None, training=False):

        batch_size = x.shape[0]
        
        assert (ids is not None) ^ (pbs is not None)
        
        input_mask = self.word_embedding.compute_mask(x)
        x = self.word_embedding(x)

        if pbs is None:
            pbs = self.pb_embedding(ids)

        if training and self.sigma > 0:
            pbs = pbs + tf.random.normal([batch_size, self.num_PB], mean=0, stddev=self.sigma)
        
        pb_dup = tf.stack([pbs]*x.shape[1], axis=1)
        
        x = tf.concat([x, pb_dup], axis=2)

        hidden = hidden if hidden is not None else self.initialize_hidden_state(batch_size)
        
        states = []
        for i, lstm in enumerate(self.lstms):
            x, h, c = lstm(x, initial_state=hidden[i], training=training, mask=input_mask)
            states.append((h, c))

        outputs = self.logits(x)
        outputs = tf.reshape(outputs, [batch_size, -1, self.vocab_size])
        
        return outputs, states, pbs


    def initialize_hidden_state(self, batch_size):
        return [(tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))) for i in range(self.num_layers)]

    
    def recognize(self, X, eps = 0.0001, early_stop_steps = 3, iters = None, step = None, max_iters=500):
        X_inputs = X[:, :-1]
        X_targets = X[:, 1:]
        X_mask = 1 - np.equal(X_targets, 0)

        batch_size = X.shape[0]

        pbs = tf.Variable(tf.zeros((batch_size, self.num_PB)))

        losses = []
        n = 0

        optimizer = tfa.optimizers.LazyAdam(self.pb_lr)
        pbs_list = []

        stagnation_steps = 0
        best_loss = float('inf')
        while n < iters if iters is not None else stagnation_steps < early_stop_steps and n < max_iters:

            with tf.GradientTape() as tape:
                X_outputs, _, _= self(X_inputs, pbs=pbs)

                X_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = X_targets, logits=X_outputs) * X_mask
                X_loss = tf.reduce_mean(tf.reduce_sum(X_loss, axis = 1))

            variables = [pbs]
            gradients = tape.gradient(X_loss, variables)
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
            optimizer.apply_gradients(zip(gradients, variables))

            if n % 100 == 0:
                print("Recognition iteration {} Loss {:.3f}".format(n, X_loss))

            if step is not None and n % step == 0:
                pbs_list.append(pb[0])

            if X_loss < best_loss - eps:
                best_loss = X_loss
                stagnation_steps = 0
            else:
                stagnation_steps += 1

            n += 1

        print("Recognition finished, iteration {} Loss {:.3f}".format(n, X_loss))

        return pbs if step is None else pbs_list

    
    def generate(self, pb, max_length, start, end):
        
        inp = tf.expand_dims([start], 0)
        pbs = tf.expand_dims(pb, 0)

        result = [start]
        hidden = None
        for t in range(max_length):

            outputs, hidden, _ = self(inp, pbs=pbs, hidden=hidden)
            preds = tf.argmax(outputs, axis = 2)
            pred = preds[0, 0].numpy()

            result.append(pred)

            if pred == end: break

            inp = tf.expand_dims([pred], 0)

        return result

    
    def get_pbs(self, ids):
        return self.pb_embedding(ids).numpy()

    
    def print_pb_stats(self):
        
        ids = tf.convert_to_tensor(list(range(self.num_sequences)))

        pbs = self.get_pbs(ids)

        centroid = np.mean(pbs, axis = 0)

        centroid_dists = np.sqrt(np.sum((pbs - np.expand_dims(centroid, 0))**2, axis=1))

        mean_dist = np.mean(centroid_dists)
        max_dist = np.max (centroid_dists)

        print("Dist origin to centroid:", np.sqrt(np.sum(centroid**2)))
        print("Mean distance from centroid:", mean_dist)
        print("Max distance from centroid:", max_dist)
    
    
    def reset_pbs(self):
        self.pb_embedding.set_weights([tf.zeros([self.num_sequences, self.num_PB])])
        
    
    def variables_expt_pb(self):
        var = [x for x in self.variables if x is not self.pb_embedding.variables[0]]
        
        assert len(var) == len(self.variables) - 1
        
        return var
