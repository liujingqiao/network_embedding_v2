import networkx as nx
from graph import Graph
from args import parse_args



if __name__ == '__main__':
    args = parse_args()
    classes_path = args.classes_input
    edges_path = args.edges_list
    graph = Graph()
    graph.load_edgelist(edges_path)