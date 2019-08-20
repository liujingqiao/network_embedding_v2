from pre_train import PreTraining
import numpy as np
import utils
import evaluate
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Embedding, Conv2D, MaxPool2D, AveragePooling2D, LSTMCell, RNN, BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.initializers import RandomNormal
from sklearn.model_selection import KFold
from scipy import stats


class Interactive:
    def __init__(self, graph, args):
        self.args = args
        self.classes_path = args.classes_input
        self.edges_path = args.edges_list
        self.batch_size = args.batch_size

        self.graph = graph
        self.emb_dim = args.emb_dim
        self.emb_size = len(graph.nodes)

        self.embedding = self.init_embedding()

    def init_embedding(self):
        embedding = dict()
        pre_train = PreTraining(self.graph, self.args)

        embedding['walk'] = pre_train.walk_proximity(trained=True)
        embedding['stru'] = pre_train.structure_proximity(trained=True)
        embedding['link'] = pre_train.link_proximity(trained=True)
        embedding['attr'] = pre_train.attributes_proximity(trained=True)
        embedding['walk_struct'] = pre_train.attributes_proximity(trained=True)
        return embedding

    def create_model(self, embedding, types, seed=6):
        vi = Input(shape=(), dtype=tf.int32)
        seed = RandomNormal(mean=0.0, stddev=0.05, seed=seed)

        # embedding信息
        walk_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['walk']])
        stru_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['stru']])
        attr_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['attr']])
        link_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['link']])
        walk_struct_emb = Embedding(self.emb_size, self.emb_dim, trainable=False, weights=[embedding['walk_struct']])

        # Connection Interaction
        concat = tf.concat([walk_emb(vi),stru_emb(vi),attr_emb(vi),link_emb(vi)], axis=1)
        attention = Dense(concat.shape[1], activation='softmax', kernel_initializer=seed)(concat)
        concat = concat*attention

        shape = 4
        # Convolutional Interaction
        reshape = tf.reshape(concat, shape=(-1, shape, self.emb_dim))
        reshape = tf.expand_dims(reshape, -1)
        inter2 = None
        for i, size in enumerate([[shape, 5], [shape, 4], [shape, 2]]):
            conv = Conv2D(filters=5, kernel_size=size, kernel_initializer=seed)(reshape)
            pool = AveragePooling2D(pool_size=(1, 2))(conv)
            dim = pool.shape[1]*pool.shape[2]*pool.shape[3]
            conv = tf.reshape(pool, shape=(-1, dim))
            if i == 0:
                inter2 = conv
            else:
                inter2 = tf.concat([inter2, conv], axis=1)

        # Merge different ways of nteraction
        #inter1 = tf.concat([walk_emb(vi), stru_emb(vi), link_emb(vi)], axis=1)
        inter1 = tf.concat([walk_emb(vi),stru_emb(vi),attr_emb(vi),link_emb(vi)], axis=1)
        attention = Dense(inter1.shape[1], activation='softmax', kernel_initializer=seed)(inter1)
        inter1 = inter1*attention

        output = tf.concat([inter1, inter2], axis=1)
        output = Dense(types, activation='softmax', kernel_initializer=seed)(output)

        model = Model(inputs=[vi], outputs=[output])

        return model

    def train(self):
        embeddings = self.init_embedding()
        data, label, types = self.graph.load_classes(self.classes_path)
        x_train, y_train, x_test, y_test = utils.train_test_split(data, label, train_size=0.7, seed=42)

        # self.folds = self.args.folds
        self.folds = 3
        kf = KFold(n_splits=self.folds)
        y_preds = np.zeros((len(x_test), self.folds))   # y_preds[i]保存第i轮的测试集预测结果
        for fold_n, (train_index, val_index) in enumerate(kf.split(y_train)):
            x_trn, y_trn = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]

            opt = Adam(0.01)
            model = self.create_model(embeddings, types, 42)
            model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy())

            patient, best_score = 0, 0
            for epoch in range(1000):
                generator = utils.batch_iter(x_trn, self.batch_size)
                for index in generator:
                    model.train_on_batch([x_trn[index]], np.eye(types)[y_trn[index]])
                y_val_pred = np.argmax(model.predict([x_val]), -1)

                micro, macro = evaluate.f1(y_val, y_val_pred)
                if micro + macro > best_score:
                    patient = 0
                    best_score = micro + macro
                    model.save_weights('dataset/output/weights/classifier')
                patient += 1
                if patient >= 50:
                    break

            # 预测
            model.load_weights('dataset/output/weights/classifier')  # 加载当前fold的最佳权重来预测test
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, -1)  # 将概率最大所在的索引作为预测结果
            y_preds[:, fold_n] = y_pred
            score = evaluate.f1(y_test, y_pred)
            print("fold_{}: {}".format(fold_n + 1, score))

        # 投票
        y_preds = stats.mode(y_preds, axis=1)[0].reshape(-1)
        score = evaluate.f1(y_test, y_preds)
        print('micro score: {}, macro score: {}'.format(score[0], score[1]))