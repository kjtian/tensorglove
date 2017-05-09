from __future__ import division
from collections import Counter, defaultdict
import os
from random import shuffle
import tensorflow as tf
import numpy as np
import pickle

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class GloVeModel():
    def __init__(self, embedding_size=30, context_size=1, max_vocab_size=100000, min_occurrences=5,
                 scaling_factor=0.8, cooccurrence_cap=100, batch_size=512, learning_rate=0.05, cooccur=None):
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__words = None
        self.__word_to_id = None
        self.__cells = None
        self.__cell_to_id = None
        self.__times = None
        self.__time_to_id = None
        self.__cooccurrence_tensor = cooccur
        self.__embeddings = None
        self.__cembeddings = None
        self.__tembeddings = None

    def fit_to_corpus(self):
        self.__word_to_id = pickle.load(open("word_dict_politics.p", 'rb')) 
        self.__words = self.__word_to_id.keys()
        self.__cells = ["Conservative", "NeutralPolitics", "PoliticalDiscussion", "SandersForPresident", "The_Donald", "politics"]
        self.__cell_to_id = {cell: i for i, cell in enumerate(self.__cells)}
        self.__times = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        self.__time_to_id = {time: i for i, time in enumerate(self.__times)}
        self.__build_graph()

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cell_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name = "cells")
            self.__time_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name = "times")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            # Embedding variables

            focal_embeddings = tf.Variable(
                tf.nn.l2_normalize(tf.random_normal([self.vocab_size, self.embedding_size]), dim=1),
                name="focal_embeddings")
            context_embeddings = focal_embeddings
            cell_embeddings = tf.Variable(
                tf.nn.l2_normalize(tf.random_normal([self.num_cells, self.embedding_size]), dim=1),
                name = "cell_embeddings")
            time_embeddings = tf.Variable(
                tf.nn.l2_normalize(tf.random_normal([self.num_times, self.embedding_size]), dim=1),
                name = "time_embeddings")

            # Bias variables

            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                       name='focal_biases')
            """
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")
            """
            context_biases = focal_biases
            cell_biases = tf.Variable(tf.random_uniform([self.num_cells], 1.0, -1.0),
                                        name="cell_biases")
            time_biases = tf.Variable(tf.random_uniform([self.num_times], 1.0, -1.0),
                                        name="time_biases")

            # Loss function

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            cell_embedding = tf.nn.embedding_lookup([cell_embeddings], self.__cell_input)
            time_embedding = tf.nn.embedding_lookup([time_embeddings], self.__time_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)
            cell_bias = tf.nn.embedding_lookup([cell_biases], self.__cell_input)
            time_bias = tf.nn.embedding_lookup([time_biases], self.__time_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.mul(tf.mul(tf.mul(tf.mul(tf.mul(focal_embedding, context_embedding), cell_embedding), cell_embedding), time_embedding), time_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                time_bias,
                cell_bias,
                focal_bias,
                context_bias,
                tf.neg(log_cooccurrences)]))

            single_losses = tf.mul(weighting_factor, distance_expr)
            self.__total_loss = tf.add_n([tf.reduce_sum(single_losses), tf.mul(0.01, tf.reduce_sum(tf.abs(cell_embedding))), tf.mul(0.01, tf.reduce_sum(tf.abs(time_embedding)))])

            self.__optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.__total_loss)

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")
            self.__cell_embeddings = cell_embeddings
            self.__time_embeddings = time_embeddings

    def train(self, num_epochs):
        batches = self.__prepare_batches()
        total_steps = 0
        with tf.Session(graph=self.__graph) as session:
            tf.initialize_all_variables().run()
            for epoch in range(num_epochs):
                print epoch
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, k_s, l_s, counts = batch
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__cell_input: k_s,
                        self.__time_input: l_s,
                        self.__cooccurrence_count: counts}
                    session.run([self.__optimizer], feed_dict=feed_dict)
                    total_steps += 1
            self.__embeddings = self.__combined_embeddings.eval()
            self.__cembeddings = self.__cell_embeddings.eval()
            self.__tembeddings = self.__time_embeddings.eval()

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cooccurrence_tensor is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], word_ids[2], word_ids[3], count)
                         for word_ids, count in self.__cooccurrence_tensor.items()]
        i_indices, j_indices, k_indices, l_indices, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, k_indices, l_indices, counts))

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def num_cells(self):
        return len(self.__cells)

    @property
    def num_times(self):
        return len(self.__times)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    @property
    def cembedding(self):
        if self.__cembeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__cembeddings

    @property
    def tembedding(self):
        if self.__tembeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__tembeddings

    def flush_embeddings(self):
        flushfile = open("vectors.txt", 'w')
        for word in self.__words:
            vec = self.embeddings[self.__word_to_id[word]]
            line = word + " " + " ".join(map(str, vec)) + "\n"
            flushfile.write(line)
        flushfile.write("filler")
        flushfile.close()
        flushfile = open("cellvectors.txt", 'w')
        for cell in self.__cells:
            vec = self.cembedding[self.__cell_to_id[cell]]
            line = cell + " " + " ".join(map(str, [abs(v) for v in vec])) + "\n"
            flushfile.write(line)
        flushfile.write("filler")
        flushfile.close()
        flushfile = open("timevectors.txt", 'w')
        for time in self.__times:
            vec = self.tembedding[self.__time_to_id[time]]
            line = time + " " + " ".join(map(str, [abs(v) for v in vec])) + "\n"
            flushfile.write(line)
        flushfile.write("filler")
        flushfile.close()

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id[word]

    def generate_tsne(self, path=None, size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)

def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def _batchify(batch_size, *sequences):
    for i in xrange(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)
