import numpy as np

def random_walk(self, adj, repeat=100, walk_length=15, types='deepwalk'):
    walk_list = []
    for _ in range(repeat):
        for i in range(len(adj)):
            cur = i
            context = [cur]
            for _ in range(walk_length - 1):
                link_list = np.argwhere(adj[cur] == 1).reshape(-1)
                if len(link_list) > 0:
                    choose = np.random.randint(len(link_list))
                    context.append(link_list[choose])
                    cur = link_list[choose]
                else:
                    context.extend([len(adj)] * (walk_length - len(context)))
            walk_list.append(list(map(str, context)))
    return walk_list