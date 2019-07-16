import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import itertools
import time
import math
import os
import json

class NMTBase:

    def __init__(self, embedding_size, units, num_layers, learning_rate, batch_size, vocab1_size, vocab2_size, max_trans_ratio,
                 gradient_clip, beam_size, working_dir, num_dev_prints, num_training_sequences, checkpoint_all = False, dropout = 0):

        self.embedding_size = embedding_size
        self.units = units
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vocab1_size = vocab1_size
        self.vocab2_size = vocab2_size
        self.max_trans_ratio = max_trans_ratio
        self.gradient_clip = gradient_clip
        self.working_dir = working_dir
        self.beam_size = beam_size
        self.num_dev_prints = num_dev_prints
        self.num_training_sequences = num_training_sequences
        self.checkpoint_all = checkpoint_all
        self.dropout = dropout


    def do_training(self, train, dev, early_stopping_steps = 0, max_epochs = 0, warm_start=False):

        epoch_it = range(max_epochs) if max_epochs is not 0 else itertools.count()

        early_stopping_steps = early_stopping_steps if early_stopping_steps is not 0 else float('inf')
        best_dev_perplexity = float('inf')
        stagnation_steps = 0

        checkpoint_path = os.path.join(self.working_dir, "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")

        checkpoint = self.get_checkp()
        checkpoint = tf.train.Checkpoint(**checkpoint)

        if warm_start:
            print("Restoring from checkpoint for warm start...")
            newest = tf.train.latest_checkpoint(checkpoint_path)
            assert newest is not None
            checkpoint.restore(newest)

        batch_losses = []
        epoch_perplexities = []
        dev_perplexities = []

        train_testing = iter(train.get_dataset(train.X, train.X_lengths, train.Y, train.Y_lengths, train.ids, 1, shuffle=False)[0])
        dev_testing = iter(dev.get_dataset(dev.X, dev.X_lengths, dev.Y, dev.Y_lengths, dev.ids, 1, shuffle=False)[0])
        
        training_start = time.time()

        epoch_state = None
        for epoch in epoch_it:
            print("\n==== Starting epoch {} ====".format(epoch + 1))

            start = time.time()
            epoch_perplexity = 0

            epoch_perplexity, batch_losses, epoch_state = self.do_epoch(train, epoch_state)

            self.handle_post_epoch()

            epoch_perplexity /= train.num_batches
            epoch_perplexities.append(float(epoch_perplexity))
            print("Finished epoch {} in {:.1f} seconds".format(epoch + 1, int(time.time() - start)))
            print("Perplexity training: {:.3f}".format(epoch_perplexity))

            # Evaluate every second epoch
            if epoch % 2 == 0:

                if self.num_dev_prints > 0:
                    print("Generating debug prints")
                    print("Train")
                    originals, translations = self.test_translate(train_testing, train.first, train.second, train.batch_size, self.num_dev_prints, beam=True)
                    for x, y in zip(originals, translations):
                        print("\t", x, " => ", y)
                    print("Dev")
                    originals, translations = self.test_translate(dev_testing, dev.first, dev.second, dev.batch_size, self.num_dev_prints, beam=True)
                    for x, y in zip(originals, translations):
                        print("\t", x, " => ", y)

                print("Measuring development set...")
                dev_perplexity = self.eval(dev)
                dev_perplexities.append(float(dev_perplexity))

                print("Perplexity dev: {:.3f}".format(dev_perplexity))


            with open(os.path.join(self.working_dir, "training_stats.json"), "w") as f:
                pass
                f.write(json.dumps({"batch_losses" : batch_losses,
                                    "epoch_perplexities" : epoch_perplexities,
                                    "dev_perplexities" : dev_perplexities,
                                    "training_time": time.time() - training_start}))

            if not (dev_perplexity < best_dev_perplexity):
                stagnation_steps += 1
            else:
                # Made improvement
                stagnation_steps = 0

            best_dev_perplexity = min(dev_perplexity, best_dev_perplexity)

            if stagnation_steps == 0 or (self.checkpoint_all and epoch % 2 == 0):
                checkpoint.save(file_prefix = checkpoint_prefix + "{:.2f}".format(dev_perplexity))

            if stagnation_steps >= early_stopping_steps:
                break

        print("Finished training in {:.2f} seconds".format(time.time() - training_start))
        return stagnation_steps >= early_stopping_steps


    def do_epoch(self, train, epoch_state = None):

        if epoch_state is None:
            print("Initializing epoch state")
            epoch_state = {'optimizer' : tfa.optimizers.LazyAdam(self.learning_rate)}

        epoch_perplexity = 0
        batch_losses = []

        for batch, (X, X_lengths, Y, Y_lengths, ids) in enumerate(train):

            X, Y = self.trim_padding(X, X_lengths, Y, Y_lengths)

            with tf.GradientTape() as tape:
                batch_loss, batch_perplexity = self.train_step(X, X_lengths, Y, Y_lengths, ids)

            variables = self.get_variables()

            gradients = tape.gradient(batch_loss, variables)

            if self.gradient_clip != 0:
                gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)

            epoch_state['optimizer'].apply_gradients(zip(gradients, variables))

            epoch_perplexity += batch_perplexity

            batch_losses.append((batch+1, float(batch_loss)))
            if batch % 100 == 0:
                print('  Batch {} Loss {:.4f}'.format(batch, batch_loss.numpy()))

        return epoch_perplexity, batch_losses, epoch_state


    def handle_post_epoch(self):
        # Provides a handle for subclasses
        pass


    def loss(self, logits, targets, target_lengths):

        mask = 1 - np.equal(targets, 0)

        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits) * mask

        loss_ = tf.reduce_sum(loss_, axis = 1)

        training_loss = tf.reduce_mean(loss_)
        perplexity = tf.reduce_mean(tf.math.exp(loss_ / target_lengths.numpy()))

        return training_loss, perplexity

    
    def eval(self, dataset):

        perplexity = 0

        for X, X_lengths, Y, Y_lengths, ids in dataset:

            X, Y = self.trim_padding(X, X_lengths, Y, Y_lengths)

            _, batch_perplexity = self.train_step(X, X_lengths, Y, Y_lengths, ids)

            perplexity += batch_perplexity

        perplexity /= dataset.num_batches

        return perplexity


    def trim_padding(self, X, X_lengths, Y, Y_lengths):

        # Padding is added on a per-dataset basis; we can trim it further per batch for efficiency.

        max_X_length = np.max(X_lengths)
        X = X[:, :max_X_length]

        max_Y_length = np.max(Y_lengths)
        Y = Y[:, :max_Y_length]

        return X, Y


    def greedy_translate(self, x, vocab2, verbose=True):

        X = tf.convert_to_tensor([x])

        dec_input = tf.convert_to_tensor([[vocab2.w2idx["<START>"]]])

        max_length = int(math.ceil(len(x) * self.max_trans_ratio))

        translation = "<START> "

        dec_hidden = self.encode(X)
        for t in range(max_length):

            preds, dec_hidden = self.decode(dec_input, dec_hidden)

            pred = tf.argmax(preds, axis = 2)

            pred = pred[0, 0].numpy()

            translation += vocab2.idx2w[pred] + " "

            if pred == vocab2.w2idx["<END>"]: break

            dec_input = tf.convert_to_tensor([[pred]])

        return translation


    def beam_translate(self, x, vocab2, verbose=True):

        X = tf.convert_to_tensor([x])

        dec_hidden = self.encode(X)
        max_length = int(math.ceil(len(x) * self.max_trans_ratio))

        beam = [(["<START>"], dec_hidden, 1)]

        def expand_beam(words, hidden, p):
            
            last_word = words[-1]

            if last_word == "<END>" or len(words) == max_length:
                return [(words, hidden, p)], False

            dec_X = tf.convert_to_tensor([[vocab2.w2idx[last_word]]])

            preds, hidden = self.decode(dec_X, hidden)

            preds = tf.nn.softmax(tf.reshape(preds[0], [-1])).numpy()

            top = tf.math.top_k(preds, k = self.beam_size, sorted=False).indices.numpy()

            expansion = [(words + [vocab2.idx2w[x]], hidden, p * preds[x]) for x in top]

            return expansion, True

        def prune_beam(beam, k):
            return sorted(beam, key = lambda b : b[2], reverse = True)[:k]


        finished = False
        while not finished:

            new_beam = []
            finished = True
            for x in beam:
                expansion, expanded = expand_beam(*x)

                finished = finished and not expanded

                new_beam.extend(expansion)

            beam = prune_beam(new_beam, k = self.beam_size)

        if verbose:
            print("Final beam:")
            for t, _, p in beam:
                print("  {:.5f}: {}".format(p, " ".join(t)))

        best = prune_beam(beam, k = 1)[0][0]

        return " ".join(best)


    def test_translate(self, data, source_vocab, target_vocab, batch_size, N, beam = True, verbose=False):
        trans = self.beam_translate if beam else self.greedy_translate

        originals = []
        translations = []
        n = 0
        for X, X_lengths, _, _, _ in data:

            for i in range(batch_size):
                x, x_length = X[i].numpy(), X_lengths[i].numpy()

                x = x[:x_length]

                t = trans(x, target_vocab, verbose=verbose)

                translations.append(t)

                originals.append(" ".join(source_vocab.idx2w[w] for w in x))

                if verbose:
                    print("Translating {}/{}".format(n+1, N))
                
                n += 1
                if n == N: break
            if n == N: break

        assert len(translations) == N

        return originals, translations


    def do_testing(self, test):

        checkpoint_path = os.path.join(self.working_dir, "checkpoints")

        checkpoint = self.get_checkp()
        checkpoint = tf.train.Checkpoint(**checkpoint)

        ckpts = os.listdir(checkpoint_path)
        ckpts.remove('checkpoint')
        ckpts = [x for x in ckpts if 'index' in x]
        ckpts = [x.rstrip(".index").lstrip("ckpt").split('-') for x in ckpts]
        ckpt_best = "ckpt" + '-'.join(min(ckpts, key = lambda x : (float(x[0]), -int(x[1]))))
        ckpt_best = os.path.join(checkpoint_path, ckpt_best)
	
        assert ckpt_best is not None
        print("\nUSING CHECKPOINT {}\n".format(ckpt_best))
        checkpoint.restore(ckpt_best)

        use_beam = self.beam_size > 0
        testing_start = time.time()
        _, translations = self.test_translate(test, test.first, test.second, test.batch_size, test.num_batches * test.batch_size, beam = use_beam, verbose=True)

        print("Finished testing in {:.2f} seconds".format(time.time() - testing_start))
        try:
            with open(os.path.join(self.working_dir, "testing_stats.json"), "w") as f:
                f.write(json.dumps({"Testing time" : time.time() - testing_start}))
        except:
            pass

        # Strip start and end tokens
        translations = [" ".join(x.strip().split(" ")[1:-1]) for x in translations]

        return translations


    def load_chkpt(self, chkpt):
        raise NotImplemented("Not implemented")


    def summarize(self):
        print("Num layers:", self.num_layers)
        print("Units per layer:", self.units)
        print("Embedding size:", self.embedding_size)
        print("Batch size:", self.batch_size)
        print("Learning rate:", self.learning_rate)
        print("Max translation ratio:", self.max_trans_ratio)
        print("Gradient clip:", self.gradient_clip)
        print("Dropout:", self.dropout)
