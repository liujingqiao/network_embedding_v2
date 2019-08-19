from graph import Graph
from args import parse_args

import multiprocessing
from walker import RandomWalker
import graph

if __name__ == '__main__':
    args = parse_args()
    classes_path = args.classes_input
    edges_path = args.edges_list
    graph = Graph()
    graph.load_edgelist(edges_path)
    walker = RandomWalker(graph, 1, 1)
    walker.preprocess_transition_probs()
    sentences = walker.simulate_walks(num_walks=80, walk_length=10)
# 实现node2vec
# 实现walk2struct