import numpy as np
import utils
from gensim.models import Word2Vec

class PreTraining:
    def __init__(self, graph, args):
        self.args = args
        self.emb_dim = args.emb_dim
        self.edges_list = args.edge_list
        self.graph = graph
        graph.load_edgelist(self.edges_list)
        self.walk = None
        self.structure = None

        self.attributes = None
        self.structure = graph.make_adj()


    def walk_embedding(self):
        walk_data = utils.random_walk(self.struct, repeat=80, walk_length=35)
        model = Word2Vec(walk_data, size=self.stru_dim, window=10, min_count=0, workers=6)
        embedding = np.zeros((self.stru_size, self.stru_dim))
        for i in range(self.stru_size):
            embedding[i] = model[str(i)]
        np.savetxt(self.walk_output, embedding)
        return embedding