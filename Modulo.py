from queue import Queue
from typing import List, Any


from networkx.algorithms import bipartite, distance_measures, approximation, cycles
from tabulate import tabulate
import networkx as nx
import scipy.linalg as la
import matplotlib as mpl
from matplotlib import pyplot as plt
import networkx.linalg.spectrum as spec

from pylab import rcParams
from networkx.drawing.nx_agraph import to_agraph
import graph_generation

# CONSIDER ROTATION

from sys import maxsize as INT_MAX
from collections import deque

N = 100200

gr = [0] * N
for i in range(N):
    gr[i] = []


# Function to add edge
def add_edge(x: int, y: int) -> None:
    global gr
    gr[x].append(y)
    gr[y].append(x)


# Function to find the length of
# the shortest cycle in the graph
def shortest_cycle(n: int) -> int:
    # To store length of the shortest cycle
    global gr
    ans = INT_MAX

    # For all vertices
    for i in range(n):

        # Make distance maximum
        dist = [int(1e9)] * n

        # Take a imaginary parent
        par = [-1] * n

        # Distance of source to source is 0
        dist[i] = 0
        q = deque()

        # Push the source element
        q.append(i)

        # Continue until queue is not empty
        while q:

            # Take the first element
            x = q[0]
            q.popleft()

            # Traverse for all it's childs

            for child in gr[x]:

                # If it is not visited yet
                if dist[child] == int(1e9):

                    # Increase distance by 1
                    dist[child] = 1 + dist[x]

                    # Change parent
                    par[child] = x

                    # Push into the queue
                    q.append(child)

                    # If it is already visited
                elif par[x] != child and par[child] != x:
                    ans = min(ans, dist[x] +
                              dist[child] + 1)

                    # If graph contains no cycle
    #clean gr
    gr = [0] * N
    for i in range(N):
        gr[i] = []

    if ans == INT_MAX:
        return -1

    # If graph contains cycle
    else:
        return ans


class Hexagon:
    neighbor_hexagons: List[Any]

    def __init__(self, p: int, label: int, init_list: list = []) -> object:
        """

        :type init_list: object
        """
        self.p = p
        self.label = label
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
                        if k[0] not in self.neighbors:
                            self.neighbors.append(k[0])  # int
                            self.neighbor_hexagons.append(vertex_dict[k[0]])  # Hexagon
                    else:  # create a new hexagon vertex
                        vertex_num = vertex_num + 1

                        h = Hexagon(p, vertex_num, c)
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
    p_list = [11]

    graph_list = []
    for i in p_list:
        vertex_dict = {}
        p = i  # modulo
        vertex_num = 0
        diG = nx.DiGraph()
        G = nx.Graph()
        sim_G = nx.DiGraph()

        start_point = Hexagon(p, vertex_num, [0, 1, 0, 0, 1, 0])
        # start_point.hexagon_img()
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
            if k < 35:
                sim_G.add_node(k)
                for n in vertex_dict[k].neighbors:
                    if 35 > n and n > k:
                        sim_G.add_edge(k, n)
            """"  
            diG.add_node(k)
            G.add_node(k)
            for n in vertex_dict[k].neighbors:
                assert isinstance(n, int)
                diG.add_edge(k, n)
                if n > k:
                    G.add_edge(k,n)
                    add_edge(k,n)
            """
        """"
        #generate gradient map
        d = {}
        for k in vertex_dict:
            if k == 1:
                d[k] = 0
            else:
                #find the nearest neighbor
                for n in vertex_dict[k].neighbors:
                    if n < k:
                        d[k] = d[n]+1
        #print(d)

        #H = nx.relabel_nodes(G, vertex_dict)
        low, *_, high = sorted(d.values())
        norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
        try:
            nx.draw_planar(G,nodelist=d,
                node_size=1000,
                node_color=[mapper.to_rgba(i)
                            for i in d.values()],
                with_labels=True,
                font_color='white')           # only works if the graph is planar
        except Exception:
             nx.draw(G,
                nodelist=d,
                node_size=1000,
                node_color=[mapper.to_rgba(i)
                            for i in d.values()],
                with_labels=True,
                font_color='white')
        
        #plt.savefig("gradient_modulo_" + str(p) + ".png")  # save as png
        #plt.show()
        """
        # a better way?

        render = to_agraph(sim_G)  # this is using graphviz. Graphviz worked better than matplotlib in this case.

        render.layout('twopi')  # this is only one of the possible layouts, I will comment on this on WeChat
        # other possible layouts: http://www.adp-gmbh.ch/misc/tools/graphviz/index.html
        render.graph_attr['label'] = "modulo_" + "p_render"
        render.draw('twopi_modulo_{}.png'.format(p))

        #
        # print(G.nodes)
        # print(G.edges)
        # plt.savefig("gradient_modulo_" + str(p) + ".png")  # save as png

        A = to_agraph(sim_G)
        A.layout('dot')
        A.draw('dot_multi_'+ str(p)+'.png')
        """
        A = nx.to_numpy_matrix(diG)
        eigvals, eigvecs = la.eig(A)
        eigvals = eigvals.real
        eigvals.sort()
        first = eigvals[-1]
        second = eigvals[-2]
        #spec.adjacency_spectrum(G)
        #girth_list = [len(c) for c in cycles.minimum_cycle_basis(G)]
        n = vertex_num + 1

        girth = shortest_cycle(n)/2

        info = [p, n, bipartite.is_bipartite(G), distance_measures.diameter(G), girth , first, second]
        info_list.append(info)
        """

    #print(tabulate(info_list, headers=["p", "vertex number", "is bipartite", "diameter", "girth", "largest eigenval", "2nd largest eigenval"]))
