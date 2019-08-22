import multiprocessing
import numpy as np
import random


from utils import alias_sample, create_alias_table


class RandomWalker:
    def __init__(self, G, p=1, q=1, types=1):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param types: choose word2vec or autoencoder
        """
        self.G = G
        self.p = p
        self.q = q
        self.types = types


    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):

        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks(self, num_walks=100, walk_length=40, workers=5):
        G = self.G
        proceeding = []
        num_nodes = int(len(G.adj) / workers)
        block = []
        for i in range(workers - 1):
            block.append((int(i * num_nodes), int((i + 1) * num_nodes)))
        block.append((int((workers - 1) * num_nodes), len(G.adj)))
        pool = multiprocessing.Pool(processes=workers)
        for i, (start, end) in enumerate(block):
            proceeding.append(pool.apply_async(self._simulate_walks, (start, end, num_walks, walk_length)))
        pool.close()
        pool.join()
        if self.types == 1:
            corpus = []
            for p in proceeding:
                corpus.extend(p.get())
            return corpus

        walk_structure = np.zeros_like(self.G.adj)
        for p in proceeding:
            walk_structure += p.get()
        walk_structure = np.where(walk_structure > 0, 1, 0)
        return  walk_structure

    def _simulate_walks(self, start, end, num_walks, walk_length,):
        walks = []
        nodes = [node for node in range(start, end)]
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(
                        walk_length=walk_length, start_node=v))

        if self.types == 1:
            for i, walk in enumerate(walks):
                walks[i] = list(map(str, walk))
            return walks

        walk_structure = np.zeros_like(self.G.adj)
        for walk in walks:
            walk_structure[walk[0]][walk[1:]] = 1
        return np.array(walk_structure)

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G.adj[v][x]  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes:
            unnormalized_probs = [G.adj[node][nbr]
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}

        for edge in G.edges:
            edge = tuple(edge)
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

