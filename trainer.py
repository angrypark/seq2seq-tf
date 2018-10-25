import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time

from seq2seq import Seq2Seq
from utils.logger import setup_logger
from utils.config import save_config


class BaseTrainer:
    def __init__(self, sess, preprocessor, data, config, summary_writer):
        self.sess = sess
        self.preprocessor = preprocessor
        self.data = data
        self.config = config
        self.summary_writer = summary_writer
        self.logger = setup_logger()
        self.preprocessor.build_preprocessor()

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(models, sess)
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, model, sess):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self, model, sess):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    
class Seq2SeqTrainer(BaseTrainer):
    def __init__(self, sess, preprocessor, data, config, summary_writer):
        super(Seq2SeqTrainer, self).__init__(sess, preprocessor, data, config, summary_writer)
        # get size of data
        self.train_size = data.train_size
        self.val_size = data.val_size
        self.batch_size = config.batch_size

        # initialize global step, epoch
        self.num_steps_per_epoch = (self.train_size - 1) // self.batch_size + 1
        self.cur_epoch = 1
        self.global_step = 1

        # for summary and logger
        self.summary_dict = dict()
        self.train_summary = "Epoch : {:2d} | Step : {:8d} | Train loss : {:.4f} "
        self.val_summary = "| Val loss : {:.4f} "
        self.generation_summary = "[Query] {}\n[Reply] {}\n[Generated] {}\n" + "-"*70
            
        # checkpoint_dir
        self.checkpoint_dir = config.checkpoint_dir
        
        # train, val iterator
        self.train_iterator = None
        self.val_iterator = None

        # for translation
        self.idx2word = self.preprocessor.vectorizer.idx2word

    def build_graph(self, name="train"):
        graph = tf.Graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        
        with graph.as_default(), tf.container(name):
            self.logger.info("Building {} graph...".format(name))
            model = Seq2Seq(self.data, self.config)
            sess = tf.Session(config=tf_config, graph=graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(model.data_iterator.initializer)
            if (self.config.checkpoint_dir) and (name == "train"):
                self.logger.info('Loading checkpoint from {}'.format(
                    self.checkpoint_dir))
                model.load(sess)
                self.global_step = model.global_step_tensor.eval(sess)
        return model, sess

    def train_step(self, model, sess):
        feed_dict = {model.dropout_keep_prob: self.config.dropout_keep_prob}
        _, loss = sess.run([model.train_step, model.loss], feed_dict=feed_dict)
        return loss

    def train_epoch(self, model, sess):
        """Not used because data size is too big"""
        self.cur_epoch += 1
        loop = tqdm(range(self.num_steps_per_epoch))
        losses = list()
        scores = list()

        for step in loop:
            loss, score = self.train_step(model, sess)
            losses.append(loss)
            scores.append(score)
        train_loss = np.mean(losses)
        train_score = np.mean(scores)

    def train(self):
        # build train, val graph 
        train_model, train_sess = self.build_graph(name="train")
        val_model, val_sess = self.build_graph(name="val")
        
        # get global step and cur epoch from loaded model, zero if there is no loaded model
        self.global_step = train_model.global_step_tensor.eval(train_sess)
        self.cur_epoch = train_model.cur_epoch_tensor.eval(train_sess)

        for epoch in range(self.cur_epoch, self.config.num_epochs + 1, 1):
            self.logger.warn("="*35 + " Epoch {} Start ! ".format(epoch) + "="*35)
            self.cur_epoch = epoch
            
            # initialize loss
            losses = list()

            for step in tqdm(range(1, self.num_steps_per_epoch+1)):
                loss = self.train_step(train_model, train_sess)
                
                # increment global step
                self.global_step += 1
                train_sess.run(train_model.increment_global_step_tensor)
                
                # add loss
                losses.append(loss)

                # summarize every 50 steps
                if self.global_step % 50 == 0:
                    self.summary_writer.summarize(self.global_step, 
                                                  summarizer="train", 
                                                  summaries_dict={"loss": np.array(loss)})

                # save model
                if self.global_step % self.config.save_every == 0:
                    train_model.save(train_sess, os.path.join(self.checkpoint_dir, "model.ckpt"))
                
                # evaluate model
                if self.global_step % self.config.evaluate_every == 0:
                    val_loss = self.val(val_model, val_sess, self.global_step)
                    train_loss = np.mean(losses)
                    self.logger.warn(self.train_summary.format(self.cur_epoch, step, train_loss) \
                                     + self.val_summary.format(val_loss))
                    # initialize loss
                    losses = list()

            # val step
            self.logger.warn("="*35 + " Epoch {} Done ! ".format(epoch) + "="*35)
            train_model.save(train_sess, os.path.join(self.checkpoint_dir, "model.ckpt"))
            val_loss = self.val(val_model, val_sess, self.global_step)
            self.logger.warn(self.val_summary.format(val_loss))
            
            # increment epoch tensor
            train_sess.run(train_model.increment_cur_epoch_tensor)

    def val(self, model, sess, global_step):
        # load latest checkpoint
        model.load(sess)
        sess.run(model.data_iterator.initializer)
        
        # initialize loss and score
        losses = list()
        val_queries, val_replies, val_generated_replies = list(), list(), list()

        # define loop
        num_batches_per_epoch = (self.data.val_size - 1) // self.batch_size + 1
        loop = tqdm(range(1, num_batches_per_epoch+1))

        for step in loop:
            feed_dict = {model.dropout_keep_prob: 1}
            queries, replies, generated_replies, loss = model.val(sess, feed_dict=feed_dict)

            queries = [" ".join([token.decode("utf-8") for token in query_tokens]) for query_tokens in queries]
            replies = [" ".join([token.decode("utf-8") for token in reply_tokens]) for reply_tokens in replies]
            generated_replies = [" ".join([token.decode("utf-8") for token in generated_reply_tokens]) \
                                 for generated_reply_tokens in generated_replies]
            
            val_queries.extend(queries)
            val_replies.extend(replies)
            val_generated_replies.extend(generated_replies)
            losses.append(loss)

        val_loss = np.mean(losses)

        # summarize val loss and score
        self.summary_writer.summarize(global_step,
                                      summarizer="val",
                                      summaries_dict={"loss": np.array(val_loss)})

        # display some generated samples
        random_indices = sorted(np.random.choice(100, 10, replace=False).tolist())

        for idx in random_indices:
            self.logger.info(self.generation_summary.format(val_queries[idx], val_replies[idx], generated_replies[idx]))

        # save as best model if it is best loss
        best_loss = float(getattr(self.config, "best_loss", 1e+5))
        if val_loss < best_loss:
            self.logger.warn("[Step {}] Saving for best loss : {:.5f} -> {:.5f}".format(global_step, best_loss, val_loss))
            model.save(sess,
                       os.path.join(self.checkpoint_dir, "best_loss", "best_loss.ckpt"))
            setattr(self.config, "best_loss", "{:.5f}".format(val_loss))
            # save best config
            setattr(self.config, "best_step", str(self.global_step))
            setattr(self.config, "best_epoch", str(self.cur_epoch))
            save_config(self.config.checkpoint_dir, self.config)
            with open(os.path.join(self.checkpoint_dir, "best_loss", "generated_result.txt"), "w") as f:
                for query, reply, generated_reply in zip(val_queries, val_replies, val_generated_replies):
                    f.write("{}\t{}\t{}\n".format(query, reply, generated_reply))
        return val_loss

    def test(self, model, sess, global_step):
        # load latest model
        model.load(sess)
        test_queries, test_replies, test_queries_lengths, \
        test_replies_lengths, test_labels = self.data.load_test_data()
        
        # flatten
        row, col, _ = np.shape(test_replies)
        test_queries_expanded = [[q]*col for q in test_queries]
        test_queries_expanded = [y for x in test_queries_expanded for y in x]
        test_queries_lengths_expanded = [[l]*col for l in test_queries_lengths]
        test_queries_lengths_expanded = [y for x in test_queries_lengths_expanded for y in x]
        test_replies = [y for x in test_replies for y in x]
        test_replies_lengths = [y for x in test_replies_lengths for y in x]
        
        feed_dict = {model.input_queries: test_queries_expanded,
                     model.input_replies: test_replies,
                     model.queries_lengths: test_queries_lengths_expanded,
                     model.replies_lengths: test_replies_lengths, 
                     model.dropout_keep_prob: 1}
        probs = model.infer(sess, feed_dict=feed_dict)
        probs = np.reshape(probs, [row, col])
        return probs, test_labels
