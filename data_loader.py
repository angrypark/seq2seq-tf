import numpy as np
import tensorflow as tf
import os

class DataGenerator:
    def __init__(self, preprocessor, config):
        # get size of train and validataion set
        self.train_size = 298554955
        with open(config.val_dir, "r") as f:
            self.val_size = sum([1 for line in f])
            
        # data config
        self.train_dir = config.train_dir
        self.val_dir = config.val_dir
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.num_epochs = config.num_epochs
        
        self.pad_shapes = {"encoder_inputs": [None], 
                           "decoder_inputs": [None], 
                           "decoder_outputs": [None],
                           "query_lengths": [None], 
                           "reply_lengths": [None]}
            
    def get_train_iterator(self, index_table):
        train_files = [os.path.join(self.train_dir, fname) 
                       for fname in sorted(os.listdir(self.train_dir)) 
                       if "validation" not in fname]
        
        train_set = tf.data.TextLineDataset(train_files)
        train_set = train_set.map(lambda line: parse_single_line(line, index_table, self.max_length), 
                                  num_parallel_calls=10)
        train_set = train_set.shuffle(buffer_size=500)
        train_set = train_set.batch(self.batch_size)
        train_set = train_set.prefetch(1)
        train_set = train_set.repeat(self.num_epochs)

        train_iterator = train_set.make_initializable_iterator()
        return train_iterator
        
    def get_val_iterator(self, index_table):
        val_set = tf.data.TextLineDataset(self.val_dir)
        val_set = val_set.map(lambda line: parse_single_line(line, index_table, self.max_length), 
                              num_parallel_calls=10)
        val_set = val_set.shuffle(buffer_size=500)
        val_set = val_set.batch(self.batch_size)
        
        val_iterator = val_set.make_initializable_iterator()
        return val_iterator
            
    def load_test_data(self):
        base_dir = "/home/angrypark/reply_matching_model/data/"
        with open(os.path.join(base_dir, "test_queries.txt"), "r") as f:
            test_queries = [line.strip() for line in f]
        with open(os.path.join(base_dir, "test_replies.txt"), "r") as f:
            replies_set = [line.strip().split("\t") for line in f]
        with open(os.path.join(base_dir, "test_labels.txt"), "r") as f:
            test_labels = [[int(y) for y in line.strip().split("\t")] for line in f]

        test_queries, test_queries_lengths = zip(*[self.preprocessor.preprocess(query)
                                                         for query in test_queries])
        test_replies = list()
        test_replies_lengths = list()
        for replies in replies_set:
            r, l = zip(*[self.preprocessor.preprocess(reply) for reply in replies])
            test_replies.append(r)
            test_replies_lengths.append(l)
        return test_queries, test_replies, test_queries_lengths, test_replies_lengths, test_labels

def split_data(data):
    _, queries, replies = zip(*[line.split('\t') for line in data])
    return queries, replies

def parse_single_line(line, index_table, max_length):
    """get single line from train set, and returns after padding and indexing
    :param line: corpus id \t query \t reply
    """
    splited = tf.string_split([line], delimiter="\t")
    encoder_input = tf.string_split([splited.values[1]], delimiter=" ").values
    decoder_input = tf.concat([["<SOS>"], tf.string_split([splited.values[2]], delimiter=" ").values], axis=0)
    decoder_output = tf.concat([tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]], axis=0) 
    
    paddings = tf.constant([[0, 0],[0, max_length]])
    padded_encoder_input = tf.slice(tf.pad([encoder_input], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
    padded_decoder_input = tf.slice(tf.pad([decoder_input], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
    padded_decoder_output = tf.slice(tf.pad([decoder_output], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
    
    query_length = tf.expand_dims(tf.to_int64(tf.minimum(max_length, tf.shape(encoder_input)[0])), -1)
    reply_length = tf.expand_dims(tf.to_int64(tf.minimum(max_length, tf.shape(decoder_input)[0])), -1)
    
    encoder_input = tf.squeeze(index_table.lookup(padded_encoder_input))
    decoder_input = tf.squeeze(index_table.lookup(padded_decoder_input))
    decoder_output = tf.squeeze(index_table.lookup(padded_decoder_output))
    
    return {"encoder_inputs": encoder_input, 
            "decoder_inputs": decoder_input, 
            "decoder_outputs": decoder_output,
            "query_lengths": query_length, 
            "reply_lengths": reply_length}
