from graph import Graph
from args import parse_args

args = parse_args()
filename = args.edges_list
graph = Graph().load_edgelist(filename)
graph.load_edgelist(filename)