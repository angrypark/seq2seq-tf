import os
import tensorflow as tf
import numpy as np
from gensim.models import FastText

from utils.utils import JamoProcessor
from model import Model

def get_embeddings(vocab_list_dir, pretrained_embed_dir, vocab_size, embed_dim):
    embedding = np.random.uniform(-1/16, 1/16, [vocab_size, embed_dim])
    if os.path.isfile(pretrained_embed_dir) & os.path.isfile(vocab_list_dir):
        with open(vocab_list_dir, "r") as f:
            vocab_list = [word.strip() for word in f if len(word)>0]
        processor = JamoProcessor()
        ft = FastText.load(pretrained_embed_dir)
        num_oov = 0
        for i, vocab in enumerate(vocab_list):
            try:
                embedding[i, :] = ft.wv[processor.word_to_jamo(vocab)]
            except:
                num_oov += 1
        print("Pre-trained embedding loaded. Number of OOV : {} / {}".format(num_oov, len(vocab_list)))
    else:
        print("No pre-trained embedding found, initialize with random distribution")

    return embedding


class Seq2Seq(Model):
    def __init__(self, data, config, mode="train"):
        super(Seq2Seq, self).__init__(data, config)
        self.mode = mode
        self.build_model()
        self.init_saver()

    def build_model(self):
        # build index table
        index_table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=self.config.vocab_list,
            num_oov_buckets=0,
            default_value=0)

        self.data_iterator = self.data.get_train_iterator(
            index_table) if self.mode == "train" else self.data.get_val_iterator(index_table)

        with tf.variable_scope("inputs"):
            # Placeholders for input, output
            input_queries, input_replies, queries_lengths, replies_lengths = self.data_iterator.get_next()
            self.input_queries = tf.placeholder_with_default(input_queries, [None, self.config.max_length],
                                                             name="input_queries")
            self.input_replies = tf.placeholder_with_default(input_replies, [None, self.config.max_length],
                                                             name="input_replies")

            self.queries_lengths = tf.placeholder_with_default(queries_lengths, [None], name="queries_length")
            self.replies_lengths = tf.placeholder_with_default(replies_lengths, [None], name="replies_length")

            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        cur_batch_length = tf.shape(self.input_queries)[0]

        # Define learning rate and optimizer
        # learning_rate = tf.train.exponential_decay(self.config.learning_rate,
        #                                            self.global_step_tensor,
        #                                            decay_steps=50000,
        #                                            decay_rate=0.96,
        #                                            staircase=True)

        self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)

        # slice SOS_token and EOS_token
        self.encoder_inputs = tf.slice(self.input_queries, [0, 1], [cur_batch_length, self.config.max_length-1])
        self.encoder_lengths = self.queries_lengths - 2
        self.decoder_inputs = tf.slice(self.input_replies, [0, 0], [cur_batch_length, self.config.max_length-1])
        self.decoder_lengths = self.replies_lengths - 1
        self.decoder_outputs = tf.slice(self.input_replies, [0, 1], [cur_batch_length, self.config.max_length])

        # sequence mask to calculate loss without pad token
        self.replies_mask = tf.sequence_mask(self.decoder_lengths, maxlen=self.config.max_length-1)

        # Embedding layer
        with tf.variable_scope("embedding"):
            embeddings = tf.Variable(get_embeddings(self.config.vocab_list,
                                                    self.config.pretrained_embed_dir,
                                                    self.config.vocab_size,
                                                    self.config.embed_dim),
                                     trainable=True,
                                     name="embeddings")
            queries_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs, name="queries_embedded")
            replies_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs, name="replies_embedded")
            queries_embedded, replies_embedded = tf.cast(queries_embedded, tf.float32), tf.cast(replies_embedded,
                                                                                                tf.float32)

        # encoding layer
        with tf.variable_scope("encoder") as vs:
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.lstm_dim)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                inputs=queries_embedded,
                                                                sequence_length=self.encoder_lengths,
                                                                time_major=False)

        with tf.variable_scope("decoder") as vs:
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.lstm_dim)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.config.lstm_dim,
                                                                       attention_states,
                                                                       memory_sequence_length=self.encoder_lengths)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               attention_layer_size=self.config.lstm_dim)

            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(replies_embedded, self.decoder_lengths, time_major=False)

            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      encoder_state,
                                                      output_layer=projection_layer)

            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            self.logits = outputs.rnn_output
            self.translations = outputs.sample_id

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_outputs, logits=self.logits)
            self.loss = tf.reduce_sum(losses * self.replies_mask) / cur_batch_length
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_step = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step_tensor)

    def val(self, sess, feed_dict=None):
        loss = sess.run(self.loss, feed_dict=feed_dict)
        # probs = sess.run(self.probs, feed_dict=feed_dict)
        return loss, None, None

    def infer(self, sess, feed_dict=None):
        return sess.run(self.positive_probs, feed_dict=feed_dict)
