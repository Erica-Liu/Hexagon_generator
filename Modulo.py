from queue import Queue
from typing import List, Any


from networkx.algorithms import bipartite, distance_measures, approximation, cycles
from tabulate import tabulate
import networkx as nx
import scipy.linalg as la

from matplotlib import pyplot as plt
import networkx.linalg.spectrum as spec
import numpy as np
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


from collections import Counter


def countDistinct(arr):
    # counter method gives dictionary of elements in list
    # with their corresponding frequency.
    # using keys() method of dictionary data structure
    # we can count distinct values in array
    return len(Counter(arr).keys())

class Hexagon:
    neighbor_hexagons: List[Any]

    def __init__(self, p: int, label: int, init_list: list = [],real_list:list = []) -> object:
        """

        :type init_list: object
        """
        self.p = p
        self.label = label
        self.fold_angles = init_list
        self.real_angles = real_list
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

                    d = self.real_angles.copy()
                    i_l = (i - 1) % 6
                    i_r = (i + 1) % 6

                    d[i_l] = d[i_l] + 2 * d[i]
                    d[i_r] = d[i_r] + 2 * d[i]
                    d[i] = -d[i]

                    c = self.fold_angles.copy()


                    c[i_l] = d[i_l] % self.p
                    c[i_r] = d[i_r] % self.p
                    c[i] = d[i] % self.p


                    # update in the vertex_dict
                    k = [key for key, val in vertex_dict.items() if
                         val.fold_angles == c]  # vertex_dict: {vertex_num: hexagon}
                    if len(k) != 0:
                        if k[0] not in self.neighbors:
                            self.neighbors.append(k[0])  # int
                            self.neighbor_hexagons.append(vertex_dict[k[0]])  # Hexagon
                    else:  # create a new hexagon vertex
                        vertex_num = vertex_num + 1

                        h = Hexagon(p, vertex_num, init_list=c,real_list=d)
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
    f = open("output2.txt", "a")
    info_list = []
    p_list = [7]

    #desired_list = [129,2,5,7,10,77,20,53,29]

    graph_list = []

    for i in range(1,2):
        vertex_dict = {}
        p = 7  # modulo
        vertex_num = 0
        diG = nx.DiGraph()
        G = nx.Graph()
        sim_G = nx.DiGraph()
        start_list = [0, 1, 0, 0, 1, 0]
        start_list = [k * i for k in start_list]
        start_point = Hexagon(p, vertex_num, start_list, start_list)
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
            """
            if k < 35:
                sim_G.add_node(k)
                for n in vertex_dict[k].neighbors:
                    if 35 > n and n > k:
                        sim_G.add_edge(k, n)
            """
            diG.add_node(k)
            G.add_node(k)
            for n in vertex_dict[k].neighbors:
                assert isinstance(n, int)
                diG.add_edge(k, n)
                if n > k:
                    G.add_edge(k,n)



        #generate gradient map

        d = {}
        for k in vertex_dict: #nodes index
            if k == 0:
                d[k] = 0
            else:
                #find the nearest neighbor
                n = min(vertex_dict[k].neighbors)
                d[k] = d[n]+1


        """
        for v in d:
            if v in desired_list:
                print("index")
                print(v)
                print("distance")
                print(d[v])

       
        
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
        
        # a better way to draw?

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
        
        A = nx.to_numpy_matrix(G)
        #print(A)
        eigvals, eigvecs = la.eig(A)
        eigvals = eigvals.real
        eigvals.sort()
        distinct_eigval_num = countDistinct(eigvals)
        first = eigvals[-1]
        second = eigvals[-2]
        #spec.adjacency_spectrum(G)

        
        cycle_9_list = []
        girth_dict = {}
        girth_len_dict = {}
        for c in cycles.minimum_cycle_basis(G):
            if len(c) == 9:
                cycle_9_list.append(c)
            
            if len(c) not in girth_dict.keys():
                girth_dict[len(c)] = c
                girth_len_dict[len(c)] = 1
            else:
                girth_len_dict[len(c)] = girth_len_dict[len(c)] + 1

        print(f"\n for starting point {i}", file=f)
        print(f"girth_len_dict:{girth_len_dict}", file=f)
        print(f"cycle_9_list: {cycle_9_list}", file=f )

        
        
        for cycle_9 in cycle_9_list:
            for v in cycle_9:
                print("index:" + str(v), file=f)
                print(f'[!]fold_angles: {vertex_dict[v].fold_angles}', file=f)
                print(f"[@]real_angles:{vertex_dict[v].real_angles}", file=f)
                print(f"neighbors:{vertex_dict[v].neighbors}", file=f)
                print(f"distance:{d[v]}\n", file=f)
        #n = vertex_num + 1
        """
        cycle_9_list = [[129, 2, 5, 7, 10, 77, 20, 53, 29], [1, 3, 4, 39, 104, 8, 14, 22, 58]]
        distance_iso_dict = {}
        for s in range(1, 7):
            print(s)
            print(f"\n##########with starting point {s} #######", file=f)
            for cycle_9 in cycle_9_list:
                print(f"cycle: {cycle_9}", file=f)
                for v in cycle_9:
                    if s != 1:
                        #do the convert
                        fold_algs = vertex_dict[v].fold_angles
                        iso_fold_algs = [(alg * s) % 7 for alg in fold_algs]
                        #check the related distance
                        for k in vertex_dict:
                            if vertex_dict[k].fold_angles == iso_fold_algs:
                                print("index:" + str(k), file=f)
                                print(f'[!]fold_angles: {vertex_dict[k].fold_angles}', file=f)
                                print(f"[@]real_angles:{vertex_dict[k].real_angles}", file=f)
                                print(f"neighbors:{vertex_dict[k].neighbors}", file=f)
                                print(f"distance:{d[k]}\n", file=f)
                                distance_iso_dict[v].append(d[k])
                    else:
                        distance_iso_dict[v] = [d[v]]
                        print("index:" + str(v), file=f)
                        print(f'[!]fold_angles: {vertex_dict[v].fold_angles}', file=f)
                        print(f"[@]real_angles:{vertex_dict[v].real_angles}", file=f)
                        print(f"neighbors:{vertex_dict[v].neighbors}", file=f)
                        print(f"distance:{d[v]}\n", file=f)

        """
        for key in distance_iso_dict:
            print(f"{key}'s distances: {distance_iso_dict[key]}", file=f)
        f.close()
        """
        #print(len(vertex_dict))

        #girth = shortest_cycle(n)/2
        #info = [p, distinct_eigval_num, first, second]
        #info = [p, n, girth_dict, girth_len_dict]
        #info = [p, n, bipartite.is_bipartite(G), distance_measures.diameter(G), girth , first, second]
        #info_list.append(info)

    #f.close()
    #print(tabulate(info_list,headers = ["p","# of dictinct eigenvalues","first","second"]))
    #print(tabulate(info_list, headers=["p", "vertex number", "is bipartite", "diameter", "girth", "largest eigenval", "2nd largest eigenval"]))
    #print(tabulate(info_list, headers=["p", "vertex number", "girth list","girth length dict"]))