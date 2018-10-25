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

        index_to_string_table = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=self.config.vocab_list,
            default_value="<UNK>")

        self.data_iterator = self.data.get_train_iterator(
            index_table) if self.mode == "train" else self.data.get_val_iterator(index_table)

        with tf.variable_scope("inputs"):
            # Placeholders for input, output
            batch = self.data_iterator.get_next()
            self.encoder_inputs = tf.placeholder_with_default(batch["encoder_inputs"], [None, None],
                                                             name="encoder_inputs")
            self.decoder_inputs = tf.placeholder_with_default(batch["decoder_inputs"], [None, None],
                                                             name="decoder_inputs")
            self.decoder_outputs = tf.placeholder_with_default(batch["decoder_outputs"], [None, None],
                                                             name="decoder_outputs")
            self.encoder_lengths = tf.to_int32(tf.placeholder_with_default(tf.squeeze(batch["query_lengths"]), [None], name="encoder_lengths"))
            self.decoder_lengths = tf.to_int32(tf.placeholder_with_default(tf.squeeze(batch["reply_lengths"]), [None], name="decoder_lengths"))
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        with tf.variable_scope("lengths"):
            self.cur_batch_length = tf.shape(self.encoder_inputs)[0]
            self.encoder_max_length = tf.shape(self.encoder_inputs)[1]
            self.decoder_max_length = tf.shape(self.decoder_inputs)[1]
            
        self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)

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
                                                               dtype=tf.float32)

        with tf.variable_scope("decoder") as vs:
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.lstm_dim)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.config.lstm_dim,
                                                                       encoder_outputs,
                                                                       memory_sequence_length=self.encoder_lengths)
            
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               attention_layer_size=self.config.lstm_dim)

            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(replies_embedded, self.decoder_lengths, time_major=False)
            
            # Decoder
            projection_layer = tf.layers.Dense(90000, use_bias=False, name="output_projection")
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      decoder_cell.zero_state(self.cur_batch_length, 
                                                                              tf.float32).clone(cell_state=encoder_state),
                                                      output_layer=projection_layer)
            self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False)
            self.logits = self.outputs.rnn_output
            generated_indices = tf.to_int64(self.outputs.sample_id)

        with tf.variable_scope("generate_replies") as vs:
            self.input_queries = index_to_string_table.lookup(self.encoder_inputs)
            self.input_replies = index_to_string_table.lookup(self.decoder_outputs)
            self.generated_replies = index_to_string_table.lookup(generated_indices)

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            # sequence mask to calculate loss without pad token
            self.replies_mask = tf.to_float(tf.sequence_mask(self.decoder_lengths, maxlen=self.decoder_max_length))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_outputs, logits=self.logits)
            self.loss = tf.reduce_mean(losses * self.replies_mask)
            gvs = self.optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_step = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step_tensor)

    def val(self, sess, feed_dict=None):
        input_queries, input_replies, generated_replies, loss = sess.run([self.input_queries, 
                                                                          self.input_replies, 
                                                                          self.generated_replies, 
                                                                          self.loss], feed_dict=feed_dict)
        return input_queries, input_replies, generated_replies, loss

    def infer(self, sess, feed_dict=None):
        return sess.run(self.positive_probs, feed_dict=feed_dict)
