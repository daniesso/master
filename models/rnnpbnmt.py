import tensorflow as tf
import numpy as np
from .nmtbase import NMTBase
from .rnnpb import RNNPB
import tensorflow_addons as tfa

class RNNPBNMT(NMTBase):

    def __init__(self, num_PB, binding_strength, pb_learning_rate, max_recog_epochs, bind_hard=False, autoencode=False, p_reset = 0, sigma = 0, **kwargs):
        
        self.num_PB = num_PB
        self.binding_strength = binding_strength
        self.pb_lr = pb_learning_rate
        self.bind_hard = bind_hard
        self.autoencode = autoencode
        self.max_recog_epochs = max_recog_epochs

        self.p_reset = p_reset
        self.sigma = sigma

        if self.autoencode: assert self.bind_hard

        super(RNNPBNMT, self).__init__(**kwargs)

        common_args = dict(embedding_size = self.embedding_size, units = self.units, 
                       num_layers = self.num_layers, num_PB = self.num_PB, 
                       num_sequences = self.num_training_sequences, pb_lr = self.pb_lr, 
                       gradient_clip = self.gradient_clip, batch_size = self.batch_size,
                       sigma = self.sigma, bind_hard = self.bind_hard, dropout = self.dropout)


        self.A = RNNPB(vocab_size = self.vocab1_size, **common_args)
        if self.autoencode:
            self.B = self.A
        else:
            self.B = RNNPB(vocab_size = self.vocab2_size, **common_args)


    def forward(self, A_X, B_X, ids=None, training=False):

        A_logits, _ , A_PB = self.A(A_X, ids=ids, training=training)

        if self.autoencode:
            B_logits, B_PB = A_logits, A_PB
        else:
            B_logits, _ , B_PB = self.B(B_X, ids=ids, training=training)

        return (A_logits, A_PB, B_logits, B_PB)


    def joint_loss(self, A_logits, A_PB, A_Y, A_lengths, B_logits, B_PB, B_Y, B_lengths):

        pred_loss_A, perplexity_A = self.loss(logits=A_logits, targets=A_Y, target_lengths=A_lengths)
        pred_loss_B, perplexity_B = self.loss(logits=B_logits, targets=B_Y, target_lengths=B_lengths)

        pred_loss = (pred_loss_A + pred_loss_B) / 2
        perplexity = (perplexity_A + perplexity_B) / 2

        if self.bind_hard:
            training_loss = pred_loss
        else:
            training_loss = pred_loss + self.binding_strength * tf.nn.l2_loss(A_PB - B_PB)

        return training_loss, perplexity


    def train_step(self, X, X_lengths, Y, Y_lengths, ids):
        A_X = X[:, :-1]
        B_X = Y[:, :-1]

        A_Y = X[:, 1:]
        B_Y = Y[:, 1:]

        A_logits, A_PB, B_logits, B_PB = self.forward(A_X, B_X, ids, training=True)

        return self.joint_loss(A_logits, A_PB, A_Y, X_lengths-1, B_logits, B_PB, B_Y, Y_lengths-1)

    def do_epoch(self, train, epoch_state = None):

        if epoch_state is None:
            print("Initializing epoch state")
            epoch_state = {'optimizer_w' : tfa.optimizers.LazyAdam(self.learning_rate),
                           'optimizer_pb' : tfa.optimizers.LazyAdam(self.pb_lr)}

        epoch_perplexity = 0
        batch_losses = []

        for batch, (X, X_lengths, Y, Y_lengths, ids) in enumerate(train):

            X, Y = self.trim_padding(X, X_lengths, Y, Y_lengths)

            with tf.GradientTape() as tape:
                batch_loss, batch_perplexity = self.train_step(X, X_lengths, Y, Y_lengths, ids)

            w_vars = self.get_variables()
            pb_vars = self.A.pb_embedding.variables + self.B.pb_embedding.variables
            w_vars = [var for var in w_vars if var is not pb_vars[0] and var is not pb_vars[1]]

            w_grads, pb_grads = tape.gradient(batch_loss, [w_vars, pb_vars])

            if self.gradient_clip != 0:
                w_grads, _ = tf.clip_by_global_norm(w_grads, self.gradient_clip)
                pb_grads, _ = tf.clip_by_global_norm(pb_grads, self.gradient_clip)

            epoch_state['optimizer_w'].apply_gradients(zip(w_grads, w_vars))
            epoch_state['optimizer_pb'].apply_gradients(zip(pb_grads, pb_vars))

            epoch_perplexity += batch_perplexity

            batch_losses.append((batch+1, float(batch_loss)))
            if batch % 100 == 0:
                print('  Batch {} Loss {:.4f}'.format(batch, batch_loss))

        return epoch_perplexity, batch_losses, epoch_state


    def handle_post_epoch(self):
        super(RNNPBNMT, self).handle_post_epoch()

        if self.p_reset > 0:
            rands = np.random.uniform(low=0.0, high=1.0, size=self.num_training_sequences)

            resets = np.where(rands < self.p_reset)[0]

            print("Resetting {} PBs".format(resets.shape[0]))

            A_pbs = self.A.pb_embedding.get_weights()[0]
            A_pbs[resets, :] = 0
            self.A.pb_embedding.set_weights([A_pbs])

            if not self.bind_hard:
                B_pbs = self.B.pb_embedding.get_weights()[0]
                B_pbs[resets, :] = 0
                self.B.pb_embedding.set_weights([B_pbs])


    def encode(self, A_X):
        return (self.B.initialize_hidden_state(batch_size=A_X.shape[0]), self.A.recognize(A_X, iters=83))


    def decode(self, dec_x, hidden):
        state, pbs = hidden

        preds, state, pbs = self.B(dec_x, pbs=pbs, hidden=state)

        hidden = (state, pbs)

        return preds, hidden


    def get_variables(self):
        return self.A.variables + self.B.variables


    def get_checkp(self):
        return dict(A=self.A, B=self.B)


    def eval(self, dataset):

        perplexity = 0
        for X, X_lengths, Y, Y_lengths, ids in dataset:

            X, Y = self.trim_padding(X, X_lengths, Y, Y_lengths)

            B_X = Y[:, :-1]
            B_Y = Y[:, 1:]

            A_pbs = self.A.recognize(X, max_iters = self.max_recog_epochs)

            B_logits, _ , _ = self.B(B_X, pbs=A_pbs)

            _, batch_perplexity = self.loss(targets=B_Y, logits=B_logits, target_lengths = Y_lengths - 1)

            perplexity += batch_perplexity

        perplexity /= dataset.num_batches

        return perplexity

    def summarize(self):
        super(RNNPBNMT, self).summarize()

        print("Num PBs:", self.num_PB)
        print("Bind hard:", self.bind_hard)
        print("Binding strength:", self.binding_strength)
        print("Autoencode:", self.autoencode)
        print("PB learning rate:", self.pb_lr)
        print("Sigma:", self.sigma)
        print("p_reset:", self.p_reset)
        print("Max recog epochs:", self.max_recog_epochs)


