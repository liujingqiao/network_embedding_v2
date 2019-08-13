import numpy as np
import utils
from gensim.models import Word2Vec


class PreTraining:
    def __init__(self, graph, args):
        self.args = args
        self.edges_path = args.edges_list
        self.hidden_size = args.hidden_size
        self.walk_embedding = args.walk_embedding

        self.graph = graph
        graph.load_edgelist(self.edges_path)
        self.adjacency = self.graph.make_adj()
        self.structure = None
        self.emb_dim = args.emb_dim
        self.emb_size = len(graph.nodes)

    def walk_training(self, repeat=100, walk_length=40, trained=False, types=1):
        """
        :param repeat: 每个顶点重复游走的次数
        :param walk_length: 顶点每次游走的长度
        :param trained: trained=false时，表示模型需要重新训练
        :param types: 使用deepwalk时type=1，使用node2vec时type=2
        :return: 返回训练结果embedding
        """
        if trained:
            return np.loadtxt(self.walk_embedding)
        corpus = None
        if types == 1:
            corpus = utils.random_walk(self.adjacency, repeat, walk_length)
        model = Word2Vec(corpus, size=self.emb_dim, window=10, min_count=0, workers=6)

        embedding = np.zeros((self.emb_size, self.emb_dim))
        for i in range(self.emb_size):
            embedding[i] = model[str(i)]
        np.savetxt(self.walk_embedding, embedding)

