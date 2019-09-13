import numpy as np
import utils
import evaluate
import tensorflow as tf
import multiprocessing
from scipy import stats
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Embedding, Lambda
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.metrics import binary_crossentropy,categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam, SGD
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from gensim.models import Word2Vec
from evaluate import Evaluate
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

    def deep_walk(self, trained=False, num_walks=100, walk_length=40, workers=5):
        """
        :param num_walks: 每个顶点重复游走的次数
        :param walk_length: 顶点每次游走的长度
        :param trained: trained=false时，表示模型需要重新训练
        :return: 返回训练结果embedding
        """
        if trained:
            return np.loadtxt(self.walk_embedding)
        # walker = RandomWalker(self.graph, p, q, types=1)
        # walker.preprocess_transition_probs()
        # sentences = walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=13)
        sentences = utils.deep_walk(self.graph.adj, num_walks=num_walks, walk_length=walk_length, workers=workers)
        print('已经游走完成..')

        model = Word2Vec(sentences, size=self.emb_dim, window=5, min_count=0, workers=13, sg=1, hs=0)

        embedding = np.zeros((self.emb_size, self.emb_dim))
        for i in range(self.emb_size):
            embedding[i] = model[str(i)]
        np.savetxt(self.walk_embedding, embedding)

    def walk_proximity(self, trained=True, num_walks=100, walk_length=40, workers=5):
        if trained:
            return np.loadtxt(self.walk_structure_embedding)
        walk_structure = utils.walk_proximity(self.graph.adj, num_walks, walk_length, workers=workers)
        print('游走已完成...')
        loss = Evaluate(10).loss()
        auto_encoder = SparseAE(self.args, walk_structure, loss, self.walk_structure_embedding)
        embedding = auto_encoder.train(parallel=False)
        return embedding

    def structure_proximity(self, trained=True):
        if trained:
            return np.loadtxt(self.args.stru_embedding)
        # pubmed:30,
        loss = Evaluate(30).loss()
        auto_encoder = SparseAE(self.args, self.adjacency, loss, self.args.stru_embedding)
        embedding = auto_encoder.train(parallel=False)
        return embedding

    def attributes_proximity(self, trained=True):
        if trained:
            return np.loadtxt(self.args.attr_embedding)
        # citeseer 20, pubmed 10,
        loss = Evaluate(20).loss()
        auto_encoder = SparseAE(self.args, self.attributes, loss, self.args.attr_embedding)
        embedding = auto_encoder.train()
        return embedding

    def link_proximity(self, trained=True):
        if trained:
            try:
                return np.loadtxt(self.args.link_embedding)
            except:
                print('文件不存在..')
                return
        link_embedding = OtherEmbeddig(self.graph, self.args, types='link')
        embedding = link_embedding.train()
        return embedding

    def classes_proximity(self, trained=True):
        if trained:
            try:
                return np.loadtxt(self.args.class_embedding)
            except:
                print('文件不存在..')
                return
        class_embedding = OtherEmbeddig(self.graph, self.args, types='classes')
        embedding = class_embedding.train()
        return embedding


class SparseAE(object):
    def __init__(self, args, data, loss, path):
        self.data = data
        self.loss = loss
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
        embedding = np.zeros((self.data_size, self.emb_dim))
        kf = KFold(n_splits=self.folds)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if parallel:

            proceeding = []
            workers = 5
            pool = multiprocessing.Pool(processes=workers)

            for fold_n, (train_index, test_index) in enumerate(kf.split(self.data)):
                x_train, x_test = self.data[train_index], self.data[test_index]
                proceeding.append(pool.apply_async(self._parallel_train, (fold_n, x_train, x_test)))

            pool.close()
            pool.join()

            for p in proceeding:
                embedding += p.get()
            embedding /= self.folds
            np.savetxt(self.embedding_path, embedding)
            return embedding

        for fold_n, (train_index, test_index) in enumerate(kf.split(self.data)):
            x_train, x_test = self.data[train_index], self.data[test_index]
            embedding += self._parallel_train(fold_n, x_train, x_test)
        embedding /= self.folds
        np.savetxt(self.embedding_path, embedding)
        return embedding

    def _parallel_train(self, fold_n, x_train, x_test):
        print('fold_n:{}'.format(fold_n))
        opt = Adam(0.01)
        self.model, self.emb = self.create_model()
        self.model.compile(optimizer=opt, loss=self.loss)
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
            if (patient > 25 and best_score > 0.7) or patient > 50:
                break
            print(score,best_score)
        print("fold_n:{}, score:{}".format(fold_n + 1, best_score))

        self.model = load_model('dataset/output/model'+str(fold_n)+'.h5', custom_objects={'loss_high_order': self.loss})
        return self.embedding(fold_n)

    def embedding(self, fold_n):
        emb = load_model('dataset/output/emb_model'+str(fold_n)+'.h5')
        embedding = emb.predict(self.data)

        return embedding

    def save_best_model(self, best_score, data, patient, fold_n):
        y_true = data.reshape(-1)
        y_pred = self.model.predict(data).reshape(-1)
        try:
            score = roc_auc_score(y_true, y_pred)
            patient += 1
            if score > best_score:
                patient = 0
                best_score = score
                self.model.save('dataset/output/model'+str(fold_n)+'.h5')
                self.emb.save('dataset/output/emb_model'+str(fold_n)+'.h5')
        except:
            score = 0
            print(y_true)

        return score, best_score, patient


class OtherEmbeddig:
    def __init__(self, graph, args, embedding=None, types='link'):
        self.graph = graph
        self.args = args
        self.embedding = embedding
        self.types = types

        self.epoch = args.epoch
        self.folds = args.folds
        self.batch_size = args.batch_size
        self.link_output = args.link_embedding

        self.emb_dim = args.emb_dim
        self.emb_size = len(graph.nodes)

    def create_model(self, classes=2):
        vi = Input(shape=(), dtype=tf.int32)
        vj = Input(shape=(), dtype=tf.int32)
        emb_layer = Embedding(self.emb_size, self.emb_dim)
        vi_emb = emb_layer(vi),
        if self.types == 'link':
            vj_emb = emb_layer(vj)
            out = Dense(classes, activation='softmax')(emb_layer(vi) * emb_layer(vj))
            model = Model(inputs=[vi, vj], outputs=[out])
        if self.types == 'classes':
            out = Dense(classes, activation='softmax')(emb_layer(vi))
            model = Model(inputs=[vi], outputs=[out])
        return model, emb_layer

    def train(self, workers=1, rate=0.5):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if self.types == 'link':
            x_train, y_train = self.graph.sampled_link(num_neg=1)
            num_class = 2
        if self.types == 'classes':
            x_train, y_train, num_class = self.graph.load_classes(self.args.classes_input)

        proceeding = []
        pool = multiprocessing.Pool(processes=workers)

        kf = KFold(n_splits=self.folds, shuffle=True)
        for fold_n, (train_index, val_index) in enumerate(kf.split(y_train)):
            x_trn, y_trn = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]
            proceeding.append(pool.apply_async(self._train, (fold_n, x_trn, y_trn, x_val, y_val, num_class)))

        pool.close()
        pool.join()
        embedding = np.zeros((len(self.graph.nodes), self.emb_dim))
        for p in proceeding:
            embedding += p.get()
        embedding /= self.folds
        if self.types == 'link':
            np.savetxt(self.link_output, embedding)
        if self.types == 'classes':
            np.savetxt(self.args.class_embedding, embedding)
        return embedding

    def _train(self, fold_n, x_trn, y_trn, x_val, y_val, num_class=2):
        # 初始化模型
        model, emb = self.create_model(num_class)
        opt = Adam(0.01)
        model.compile(optimizer=opt, loss=categorical_crossentropy)

        patient, best_score = 0, 0
        best_embedding = None
        for epoch in range(2000):
            generator = utils.batch_iter(x_trn, self.batch_size)
            for index in generator:
                if self.types == 'classes':
                    model.train_on_batch([x_trn[index]], np.eye(num_class)[y_trn[index]])
                if self.types == 'link':
                    vi, vj = x_trn[index][:, 0], x_trn[index][:, 1]
                    model.train_on_batch([vi, vj], np.eye(num_class)[y_trn[index].reshape(-1).astype(int)])

            if self.types == 'classes':
                y_val_pred = np.argmax(model.predict([x_val]), -1)
                micro, macro = Evaluate.f1(y_val, y_val_pred)
                print('fold_{}:,{},{}'.format(fold_n, micro, macro))
                score = micro+macro
            if self.types == 'link':
                y_val_pred = np.argmax(model.predict([x_val[:, 0], x_val[:, 1]]), -1)
                score = roc_auc_score(y_val, y_val_pred)
                print('fold_{}:,{},{}'.format(fold_n, score, best_score))

            if score > best_score:
                patient = 0
                best_score = score
                best_embedding = emb.get_weights()[0]
            patient += 1
            if patient >= 50:
                break
        return best_embedding
