from queue import Queue
from typing import List, Any
from networkx.algorithms import bipartite, distance_measures, approximation
from tabulate import tabulate

import matplotlib.pyplot as plt
import networkx as nx


class Hexagon:
    neighbor_hexagons: List[Any]

    def __init__(self, p: object, init_list: object = []) -> object:
        self.p = p
        self.fold_angles = init_list
        self.neighbors = []
        self.neighbor_hexagons = []
        self.generated = False

    def neighbor_gen(self):
        global vertex_num
        global vertex_dict
        if not self.generated:
            self.generated = True
            for i in range(6):
                if self.fold_angles[i] > 0:
                    c = self.fold_angles.copy()
                    i_l = (i - 1) % 6
                    i_r = (i + 1) % 6
                    c[i_l] = (c[i_l] + 2 * c[i]) % self.p
                    c[i_r] = (c[i_r] + 2 * c[i]) % self.p
                    c[i] = (-c[i]) % self.p
                    # update in the vertex_dict
                    k = [key for key, val in vertex_dict.items() if
                         val.fold_angles == c]  # vertex_dict: {vertex_num: hexagon}
                    if len(k) != 0:
                        self.neighbors.append(k[0])  # int
                        self.neighbor_hexagons.append(vertex_dict[k[0]])  # Hexagon
                    else:  # create a new hexagon vertex
                        vertex_num = vertex_num + 1
                        h = Hexagon(p, c)
                        vertex_dict[vertex_num] = h
                        self.neighbors.append(vertex_num)
                        self.neighbor_hexagons.append(h)
        return self.neighbor_hexagons

    def hexagon_img(self):

        g = nx.generators.lattice.hexagonal_lattice_graph(1, 1)
        angle_dict = {i: self.fold_angles[i] for i in range(6)}
        g_1 = nx.relabel_nodes(g, angle_dict)
        nx.draw(g_1)
        plt.savefig("start_point.png")  # save as png
        plt.show()  # display


if __name__ == '__main__':

    info_list = []
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        vertex_dict = {}
        p = i  # modulo
        vertex_num = 0
        G = nx.Graph()

        start_point = Hexagon(p, [0, 1, 0, 0, 1, 0])
        # start_point.hexagon_img()
        vertex_num = 1
        vertex_dict[vertex_num] = start_point
        q = Queue()
        q.put(start_point)
        while not q.empty():
            t = q.get()
            neighbors_generated = t.neighbor_gen()
            for n in neighbors_generated:
                if not n.generated:
                    q.put(n)

        for k in vertex_dict:
            G.add_node(k)
            for n in vertex_dict[k].neighbors:
                assert isinstance(n, int)
                G.add_edge(k, n)

        H = nx.relabel_nodes(G, vertex_dict)
        nx.draw(H)
        plt.savefig("modulo_" + str(p) + ".png")  # save as png
        plt.show()  # display
        info = [p, vertex_num, nx.is_connected(G), bipartite.is_bipartite(G), distance_measures.diameter(G)]
        info_list.append(info)

    print(tabulate(info_list, headers=["p", "vertex number", "is connected", "is bipartite", "diameter"]))
