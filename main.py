from graph import Graph
from args import parse_args
from pre_train import PreTraining
from interactive import Interactive


def train(walk=False, struc=False, link=False, attr=False):
    pre_train = PreTraining(graph, args)
    pre_train.walk_proximity(trained=walk, repeat=200, walk_length=50)
    pre_train.structure_proximity(trained=struc)
    pre_train.link_proximity(trained=link)
    pre_train.attributes_proximity(trained=attr)

if __name__ == '__main__':
    args = parse_args()
    classes_path = args.classes_input
    edges_path = args.edges_list
    graph = Graph()
    graph.load_edgelist(edges_path)
    train(walk=False, struc=True, link=True, attr=True)
    interactive = Interactive(graph, args)
    interactive.train()
