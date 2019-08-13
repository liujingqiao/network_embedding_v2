import numpy as np


class Graph:
    def __init__(self, is_directed=True):
        self.is_directed = is_directed
        self.edges = None
        self.nodes = None
        self.attributes = None
        self.node2idx = None
        self.idx2node = None

    def indexed_node(self):
        # indexed node
        node2idx, idx2node = {}, {}
        for i, node in enumerate(self.nodes):
            if node not in node2idx:
                node2idx[node] = i
                idx2node[i] = node
                i = i + 1
        return node2idx, idx2node

    def load_edgelist(self, filename):
        edges, nodes = [], set()
        with open(filename, 'r') as fr:
            for line in fr:
                line = line.split()
                edges.append(line)
                nodes = nodes.union(line)
        self.nodes = np.array(list(nodes))
        self.edges = np.array(edges)
        self.indexed_node()
        return self

    def load_classes(self, path):
        classes2idx = {}
        node, classes = [], []
        with open(path, 'r') as fr:
            for line in fr:
                line = line.split()
                classes2idx[line[1]] = classes2idx.get(line[1], len(classes2idx))
                node.append(self.node2idx[line[0]])
                classes.append(classes2idx[line[1]])
        types = len(set(classes))
        return np.array(node), np.array(classes), types

    def make_adj(self):
        # Finding Adjacency Matrix
        node_num = len(self.node2idx)
        adj = np.zeros((node_num, node_num))
        lap = np.zeros_like(adj)
        for e0, e1 in self.edges:
            adj[self.node2idx[e1]][self.node2idx[e0]] = 1
        if not self.is_directed:
            for e0, e1 in self.edges:
                adj[self.node2idx[e0]][self.node2idx[e1]] = 1
        return adj

    def load_attributes(self, filename, types=1):
        # 数据格式不同处理方式不同
        if types == 1:
            with open(filename, "r") as fr:
                line = fr.readline().split()
                dim = len(line[1:])
            attrs = np.zeros((len(self.node2idx), dim))
            with open(filename, 'r') as fr:
                for line in fr:
                    line = line.split()
                    node, attr = self.node2idx[line[0]], list(map(int, line[1:]))
                    attrs[node] = attr
            return attrs

        node_attr, attr2idx, i = {}, {}, 0
        with open(filename, 'r') as fr:
            for line in fr:
                line = line.split()
                for attr in line[1:]:
                    if attr not in attr2idx:
                        attr2idx[attr] = i
                        i += 1
                node_attr[self.node2idx[line[0]]] = [attr2idx[attr] for attr in line[1:]]

        node_attr_matrix = np.zeros((len(self.node2idx), len(attr2idx)))
        for item in node_attr.items():
            for attr in item[0]:
                node_attr_matrix[item[0]][attr] = 1
        self.attributes = node_attr_matrix
        return self


