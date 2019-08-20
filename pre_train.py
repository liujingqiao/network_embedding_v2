import numpy as np
import utils
import evaluate
import tensorflow as tf
import multiprocessing
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Embedding, Lambda
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
from walker import RandomWalker
import os

class PreTraining:
    def __init__(self, graph, args):
        self.args = args
        self.edges_path = args.edges_list
        self.attr_path = args.attr_input
        self.hidden_size = args.hidden_size
        self.walk_embedding = args.walk_embedding
        self.walk_structure_embedding = args.walk_structure_embedding

        self.graph = graph
        graph.load_edgelist(self.edges_path)
        self.adjacency = graph.adj
        self.structure = None
        self.emb_dim = args.emb_dim
        self.emb_size = len(graph.nodes)

        graph.load_attribute(self.attr_path, types=1)
        self.attributes = graph.attributes

    def walk_proximity(self, num_walks=100, walk_length=40, trained=False, p=1, q=1):
        """
        :param repeat: 每个顶点重复游走的次数
        :param walk_length: 顶点每次游走的长度
        :param trained: trained=false时，表示模型需要重新训练
        :param types: 使用deepwalk时type=1，使用node2vec时type=2
        :return: 返回训练结果embedding
        """
        if trained:
            return np.loadtxt(self.walk_embedding)
        walker = RandomWalker(self.graph, p, q, types=1)
        walker.preprocess_transition_probs()
        sentences = walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=13)
        print('已经游走完成..')

        model = Word2Vec(sentences, size=self.emb_dim, window=5, min_count=0, workers=13)

        embedding = np.zeros((self.emb_size, self.emb_dim))
        for i in range(self.emb_size):
            embedding[i] = model[str(i)]
        np.savetxt(self.walk_embedding, embedding)

    def walk_structure_proximity(self, trained=True, num_walks=100, walk_length=40, p=1, q=1):
        if trained:
            return np.loadtxt(self.walk_structure_embedding)
        walker = RandomWalker(self.graph, p, q, types=2)
        walker.preprocess_transition_probs()
        walk_structure = walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=13)
        auto_encoder = AutoEncoder(self.args, walk_structure, self.walk_structure_embedding)
        embedding = auto_encoder.train()
        np.savetxt(self.walk_structure_embedding, embedding)

    def structure_proximity(self, trained=True):
        if trained:
            return np.loadtxt(self.args.stru_embedding)
        auto_encoder = AutoEncoder(self.args, self.adjacency, self.args.stru_embedding)
        embedding = auto_encoder.train()
        return embedding

    def attributes_proximity(self, trained=True):
        if trained:
            return np.loadtxt(self.args.attr_embedding)
        auto_encoder = AutoEncoder(self.args, self.attributes, self.args.attr_embedding)
        embedding = auto_encoder.train()
        return embedding

    def link_proximity(self, trained=True):
        if trained:
            return np.loadtxt(self.args.link_embedding)
        link_embedding = LinkEmbeddig(self.graph, self.args)
        embedding = link_embedding.train()
        return embedding


class AutoEncoder(object):
    def __init__(self, args, data, path):
        self.data = data
        self.embedding_path = path

        self.data_size = data.shape[0]
        self.data_dim = data.shape[1]
        self.hidden_size = args.hidden_size
        self.emb_dim = args.hidden_size[-1]

        self.epoch = args.epoch
        self.folds = args.folds
        self.batch_size = args.batch_size

    def create_model(self):
        X = Input(shape=(self.data_dim,))

        hidden = X
        for size in self.hidden_size[:-1]:
            hidden = Dense(size, activation='relu')(hidden)
        hidden = Dense(self.hidden_size[-1], activation='relu', name='emb')(hidden)
        Y = hidden
        # decode
        for size in reversed(self.hidden_size[:-1]):
            hidden = Dense(size, activation='relu')(hidden)
        X_ = Dense(self.data_dim, activation='sigmoid')(hidden)

        model = Model(inputs=X, outputs=X_)
        encode = Model(inputs=X, outputs=Y)
        return model, encode

    def train(self, parallel=True):
        if parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            embedding = np.zeros((self.data_size, self.emb_dim))
            kf = KFold(n_splits=self.folds)

            proceeding = []
            workers = self.folds
            pool = multiprocessing.Pool(processes=workers)

            for fold_n, (train_index, test_index) in enumerate(kf.split(self.data)):
                x_train, x_test = self.data[train_index], self.data[test_index]
                proceeding.append(pool.apply_async(self._parallel_train, (fold_n, x_train, x_test)))

            pool.close()
            pool.join()
            embedding = np.zeros((self.data_size, self.emb_dim))

            for p in proceeding:
                embedding += p.get()
            embedding /= self.folds
            np.savetxt(self.embedding_path, embedding)
            return embedding

    def _parallel_train(self, fold_n, x_train, x_test):
        print('fold_n:{}'.format(fold_n))
        opt = Adam(0.01)
        self.model, self.emb = self.create_model()
        self.model.compile(optimizer=opt, loss=evaluate.loss_autoencoder)
        # self.model.compile(optimizer=opt, loss='binary_crossentropy')
        patient = 0
        best_score = 0

        for epoch in range(self.epoch):
            # batch
            generator = utils.batch_iter(x_train, self.batch_size, 1)
            for index in generator:
                self.model.train_on_batch(x_train[index], x_train[index])

            # save best reconsitution model and embedding model
            score, best_score, patient = self.save_best_model(best_score, x_test, patient, fold_n)
            if (patient > 25 and best_score > 0.7) or patient > 100:
                break
        print("fold_n:{}, score:{}".format(fold_n + 1, best_score))

        self.model = load_model('dataset/output/model'+str(fold_n)+'.h5', custom_objects={'loss_autoencoder': evaluate.loss_autoencoder})
        return self.embedding(fold_n)

    def embedding(self, fold_n):
        emb = load_model('dataset/output/emb_model'+str(fold_n)+'.h5')
        embedding = emb.predict(self.data)

        return embedding

    def save_best_model(self, best_score, data, patient, fold_n):
        y_true = data.reshape(-1)
        y_pred = self.model.predict(data).reshape(-1)

        score = roc_auc_score(y_true, y_pred)
        patient += 1
        if score > best_score:
            patient = 0
            best_score = score
            self.model.save('dataset/output/model'+str(fold_n)+'.h5')
            self.emb.save('dataset/output/emb_model'+str(fold_n)+'.h5')

        return score, best_score, patient

    # def train(self):
    #     test_pred = np.zeros_like(self.data)    # 保存训练集五次交叉预测的预测结果
    #     embedding = np.zeros((self.data_size, self.emb_dim))
    #     # train
    #     kf = KFold(n_splits=self.folds)
    #     for fold_n, (train_index, test_index) in enumerate(kf.split(self.data)):
    #         print("fold: {}".format(fold_n + 1))
    #         opt = Adam(0.01)
    #         self.model, self.emb = self.create_model()
    #         #self.model.compile(optimizer=opt, loss=evaluate.loss_autoencoder)
    #         self.model.compile(optimizer=opt, loss='binary_crossentropy')
    #         patient = 0
    #         best_score = 0
    #         x_train, x_test = self.data[train_index], self.data[test_index]
    #
    #         for epoch in range(self.epoch):
    #             # batch
    #             generator = utils.batch_iter(x_train, self.batch_size, 1)
    #             for index in generator:
    #                 self.model.train_on_batch(x_train[index], x_train[index])
    #
    #             # save best reconsitution model and embedding model
    #             score, best_score, patient = self.save_best_model(best_score, x_test, patient)
    #             if (patient > 25 and best_score > 0.7) or patient > 100:
    #                 break
    #             print("epoch:{}, score:{}".format(epoch + 1, score))
    #
    #         self.model = load_model('dataset/output/model.h5', custom_objects={'loss_autoencoder': evaluate.loss_autoencoder})
    #         test_pred[test_index] += self.model.predict(x_test)
    #         embedding += self.embedding()
    #
    #     embedding /= self.folds
    #     np.savetxt(self.embedding_path, embedding)
    #
    #     # evaluate
    #     evaluate.recontruction(test_pred, self.data)


class LinkEmbeddig:
    def __init__(self, graph, args):
        self.graph = graph
        self.args = args

        self.epoch = args.epoch
        self.folds = args.folds
        self.batch_size = args.batch_size
        self.link_output = args.link_embedding

        self.emb_dim = args.emb_dim
        self.emb_size = len(graph.nodes)

    def create_model(self):
        vi = Input(shape=(), dtype=tf.int32)
        vj = Input(shape=(), dtype=tf.int32)
        link_emb = Embedding(self.emb_size, self.emb_dim)
        vi_emb, vj_emb = link_emb(vi), link_emb(vj)
        #out = K.sum(vi_emb * vj_emb)

        out = Dense(1)(vi_emb * vj_emb)
        model = Model(inputs=[vi, vj], outputs=[out])
        return model, link_emb

    def train(self):
        x_train, y_train = self.graph.sampled_link(num_neg=3)
        embedding = np.zeros((len(self.graph.nodes), self.emb_dim))
        kf = KFold(n_splits=self.folds, shuffle=True)

        workers = self.folds
        proceeding = []
        pool = multiprocessing.Pool(processes=workers)

        for fold_n, (train_index, val_index) in enumerate(kf.split(y_train)):
            x_trn, y_trn = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]
            proceeding.append(pool.apply_async(self._train, (fold_n, x_trn, y_trn, x_val, y_val)))

        pool.close()
        pool.join()
        embedding = np.zeros((len(self.graph.nodes), self.emb_dim))
        for p in proceeding:
            embedding += p.get()
        embedding /= self.folds
        np.savetxt(self.link_output, embedding)
        return embedding

    def _train(self, fold_n, x_trn, y_trn, x_val, y_val):

        # 初始化模型
        model, emb = self.create_model()
        opt = Adam(0.01)
        model.compile(optimizer=opt, loss=evaluate.first_order_loss)

        patient, best_score = 0, 100000
        best_embedding = None
        for epoch in range(2000):
            generator = utils.batch_iter(x_trn, self.batch_size)
            for index in generator:
                vi, vj = x_trn[index][:, 0], x_trn[index][:, 1]
                model.train_on_batch([vi, vj], [y_trn[index]])
            y_pred = model.predict([x_val[:, 0], x_val[:, 1]])
            score = mean_squared_error(y_val, y_pred)
            if score < best_score:
                patient = 0
                best_score = score
                # model.save_weights('../data/output/weights/link')
                best_embedding = emb.get_weights()[0]
            patient += 1
            if patient >= 30:
                break
        print('{}:{}'.format(fold_n,best_score))
        return best_embedding


    # def _train(self):
    #     x_train, y_train = self.graph.sampled_link(num_neg=3)
    #
    #     embedding = np.zeros((len(self.graph.nodes), self.emb_dim))
    #     kf = KFold(n_splits=self.folds, shuffle=True)
    #
    #     for fold_n, (train_index, val_index) in enumerate(kf.split(y_train)):
    #         print((fold_n))
    #         # 初始化模型
    #         model, emb = self.create_model()
    #         opt = Adam(0.01)
    #         model.compile(optimizer=opt, loss=evaluate.first_order_loss)
    #         # 划分训练集和测试集
    #         x_trn, y_trn = x_train[train_index], y_train[train_index]
    #         x_val, y_val = x_train[val_index], y_train[val_index]
    #         # 开始训练
    #         patient, best_score = 0, 100000
    #         for epoch in range(2000):
    #             generator = utils.batch_iter(x_trn, self.batch_size)
    #             for index in generator:
    #                 vi, vj = x_trn[index][:, 0], x_trn[index][:, 1]
    #                 model.train_on_batch([vi, vj], [y_trn[index]])
    #             y_pred = model.predict([x_val[:, 0], x_val[:, 1]])
    #             score = mean_squared_error(y_val, y_pred)
    #             print('{}:{}'.format(epoch,score))
    #             if score < best_score:
    #                 patient = 0
    #                 best_score = score
    #                 # model.save_weights('../data/output/weights/link')
    #                 best_embedding = emb.get_weights()[0]
    #             patient += 1
    #             if patient >= 30:
    #                 break
    #
    #         embedding += best_embedding
    #     embedding /= self.folds
    #     np.savetxt(self.link_output, embedding)
    #     return embedding