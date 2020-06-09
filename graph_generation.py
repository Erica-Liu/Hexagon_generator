# By Gabriel Luo, Jan 2020.
# For Math 389 project on Monomial Relation
# Team with Alex Vidinas, Erica Liu
# Supervised by Prof. Lagarias, Prof. Boland

import matplotlib.pyplot as plt
from pathlib import Path
import functools as fp
from typing import List
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph


def graph_generation(
        starting_graph: nx.DiGraph,
        iteration: int
) -> List[nx.DiGraph]:
    resulting_graphs = list()
    print("starting with graph:")
    for e in starting_graph.edges:
        print("{} -> {}".format(e[0], e[1]))
    current_graph = starting_graph
    resulting_graphs.append(current_graph)
    for i in range(iteration):
        next_graph = nx.DiGraph()
        for v in current_graph.nodes:
            # print("for the vertex {}".format(v))
            for pred in current_graph.predecessors(v):
                for succ in current_graph.successors(v):
                    new_source_v = pred[0] + v
                    new_target_v = v + succ[-1]
                    # print("    from {} to {}".format(new_source_v, new_target_v))
                    next_graph.add_edge(new_source_v, new_target_v, label=new_source_v[0] + new_target_v)
        for e in current_graph.edges:
            next_graph.add_node(e[0][0] + e[1])
        resulting_graphs.append(next_graph)
        current_graph = next_graph

    return resulting_graphs


def show_graph(graph_list: List[nx.DiGraph], path: str, name: str):
    count = 1
    size_list = []
    Path("./graph_{}".format(path)).mkdir(parents=True, exist_ok=True)
    for graph in graph_list:
        this_size = str(len(graph.nodes))
        size_list.append(this_size)
        try:
            nx.draw_planar(graph)           # only works if the graph is planar
        except Exception:
            nx.draw(graph)                  # if the previous failed, draw nontheless
        # plt.show()
        render = to_agraph(graph)           # this is using graphviz. Graphviz worked better than matplotlib in this case. 

        render.layout('dot')                # this is only one of the possible layouts, I will comment on this on WeChat
                                            # other possible layouts: http://www.adp-gmbh.ch/misc/tools/graphviz/index.html
        render.graph_attr['label'] = name + "\nnumber of vertices = " + this_size + \
                                     "\nHilbert Function result, so far: [" + fp.reduce(
            lambda a, b: str(a) + ', ' + str(b), size_list) + "]"
        render.draw('graph_{}/graph_{}.png'.format(path, count))
        count = count + 1


def print_from_list(edge_list, functor, num_of_iter, path, name):
    # example 10, bouncing back and forth
    starting_graph = nx.DiGraph()
    for e in edge_list:
        starting_graph.add_edge(e[0], e[1], label=e[0][0] + e[1])
    final_list = graph_generation(starting_graph, num_of_iter)
    functor(final_list, path, name)


def print_graph_as_text(graph_list: List[nx.DiGraph]):
    count = 1
    graph_size_list = []
    for graph in graph_list:
        print("Gamma_{} size = {}; {}".format(count, len(graph.nodes), graph.nodes))
        graph_size_list.append(len(graph.nodes))
        count = count + 1
    for x, y in zip(graph_size_list[:-1:], graph_size_list[1::]):
        print("{} / {} = {}".format(y, x, y / x))


if __name__ == '__main__':
    # the monomial orbital starting with:
    # m1 ---> m2
    # m1 <--- m2
    # m1 <--- m3
    # m3 <--- m2
    # starting_graph1 = nx.DiGraph()
    # starting_graph1.add_edge('1', '2')
    # starting_graph1.add_edge('2', '1')
    # starting_graph1.add_edge('3', '1')
    # starting_graph1.add_edge('2', '3')
    # final_list_1 = graph_generation(starting_graph1, 20)
    # show_graph(final_list_1)

    # example 2, a variation ????
    # starting_graph2 = nx.DiGraph()
    # starting_graph2.add_edge('1', '2')
    # starting_graph2.add_edge('2', '2')
    # starting_graph2.add_edge('2', '3')
    # starting_graph2.add_edge('3', '1')
    # final_list_2 = graph_generation(starting_graph2, 10)
    # show_graph(final_list_2)

    # example 3, Fibonacci as a difference sequence
    # starting_graph3 = nx.DiGraph()
    # starting_graph3.add_edge('1', '1')
    # starting_graph3.add_edge('1', '2')
    # starting_graph3.add_edge('2', '1')
    # starting_graph3.add_edge('1', '3')
    # final_list_3 = graph_generation(starting_graph3, 10)
    # show_graph(final_list_3)

    # example 4, Fibonacci
    # starting_graph4 = nx.DiGraph()
    # starting_graph4.add_edge('1', '1')
    # starting_graph4.add_edge('1', '2')
    # starting_graph4.add_edge('2', '1')
    # final_list_4 = graph_generation(starting_graph4, 10)
    # show_graph(final_list_4)

    # example 5, linear as a difference sequence (quadratic)
    # starting_graph5 = nx.DiGraph()
    # edge_list5 = ('1', '1'), ('1', '2'), ('2', '2'), ('2', '3'), ('3', '3')
    # final_list_5 = graph_generation(starting_graph5, 10)
    # show_graph(final_list_5)
    # print_from_list(
    #     edge_list5,
    #     show_graph,
    #     path="quadratic",
    #     name="Graph that has three self-loops on a chain."
    # )

    # example 6, cubic
    # starting_graph6 = nx.DiGraph()
    # starting_graph6.add_edge('1', '1')
    # starting_graph6.add_edge('1', '2')
    # starting_graph6.add_edge('2', '2')
    # starting_graph6.add_edge('2', '3')
    # starting_graph6.add_edge('3', '3')
    # starting_graph6.add_edge('3', '4')
    # starting_graph6.add_edge('4', '4')
    # final_list_6 = graph_generation(starting_graph6, 10)
    # show_graph(final_list_6)

    # example 7, a self loop next to a 3-cycle
    # starting_graph7 = nx.DiGraph()
    # starting_graph7.add_edge('1', '1')
    # starting_graph7.add_edge('1', '2')
    # starting_graph7.add_edge('2', '3')
    # starting_graph7.add_edge('3', '1')
    # final_list_7 = graph_generation(starting_graph7, 10)
    # show_graph(final_list_7)

    # example 8, Feb 11th 2020 Suspected to be linear???
    # edge_list8 = [('1', '2'), ('2', '3'), ('3', '1'), ('3', '4'), ('5', '3'), ('4', '5'), ('5', '4')]
    # print_from_list(edge_list=edge_list8, functor=show_graph, path="unknown", name="unknown")
    # print_graph_as_text(final_list_8)

    # example 9 quadratic growth, debugging graph outputs.
    # starting_graph9 = nx.DiGraph()
    # edge_list9 = [('1', '1'), ('1', '2'), ('2', '3'), ('2', '4'), ('3', '2'), ('4', '5'), ('5', '5')]
    # starting_graph9.add_edges_from(edge_list9)
    # final_list_9 = graph_generation(starting_graph9, 10)
    # print_graph_as_text(final_list_9)
    # show_graph(final_list_9)

    # example 10, bouncing back and forth
    # edge_list10 = [('1', '2'), ('3', '1'), ('2', '3'), ('4', '2'), ('2', '5')]
    # print_from_list(
    #     edge_list10,
    #     show_graph,
    #     path="bouncing_5_6",
    #     name="Graph from A = {1, 2, 3, 4, 5}, starting with one cycle and some branches."
    # )

    # example 11, exponential
    # edge_list11 = [('ab', 'ba'), ('ab', 'bb'), ('bb', 'ba'), ('ba', 'ab')]
    # print_from_list(
    #     edge_list11,
    #     show_graph,
    #     path="exp_motivating",
    #     name="Graph from A = {a, b}, S = { aa, bbb }; motivating example."
    # )

    # example 12, linear
    # edge_list12 = [('1', '1'), ('1', '2'), ('2', '2')]
    # print_from_list(
    #     edge_list12,
    #     show_graph,
    #     path="linear",
    #     name="Graph from A = {1, 2}, S = { 21 }"
    # )

    # example 13, vanishing
    # edge_list13 = [('0', '1'), ('1', '2'), ('3', '2'), ('2', '4'), ('4', '5'), ('6', '7'), ('7', '4'), ('8', '4')]
    # # print_from_list(edge_list13, show_graph, name=" Graph that will vanish. ")
    # print_from_list(
    #     edge_list13,
    #     show_graph,
    #     path="vanishing",
    #     name="Graph that originally is a tree."
    # )

    # example 14, smallest exp
    # edge_list14 = [('1', '1'), ('1', '2'), ('2', '1')]
    # print_from_list(
    #     edge_list14,
    #     show_graph,
    #     path="small_exp",
    #     name="Graph from: A = { 1, 2 }, S = { 22 }"
    # )

    ''' Actually used examples '''
    # print_from_list(
    #     [('1', '1'), ('1', '2'), ('2', '2'), ('2', '1')],
    #     show_graph,
    #     5,
    #     path="exp1",
    #     name="Exponential, two self-loops joined by a 2-cycle"
    # )
    #
    # print_from_list(
    #     [('1', '2'), ('2', '3'), ('2', '4'), ('3', '1'), ('4', '1')],
    #     show_graph,
    #     7,
    #     path="exp2",
    #     name="Exponential, touching 3-cycles"
    # )
    #
    # print_from_list(
    #     [('1', '2'), ('2', '3'), ('3', '1'), ('3', '4'), ('2', '4'), ('6', '3'), ('4', '5'), ('5', '6'), ('6', '4')],
    #     show_graph,
    #     7,
    #     path="exp3",
    #     name="Exponential, two cycles with multiple edges in-between"
    # )
    #
    # print_from_list(
    #     [('1', '1'), ('1', '2'), ('2', '2')],
    #     show_graph,
    #     10,
    #     path="linear1",
    #     name="Linear, simplest case"
    # )
    #
    # print_from_list(
    #     [('1', '1'), ('1', '2'), ('2', '3'), ('3', '2')],
    #     show_graph,
    #     10,
    #     path="linear2",
    #     name="Linear, with a self-loop and a 2-cycle"
    # )
    #
    # print_from_list(
    #     [('1', '1'), ('1', '2'), ('2', '2'), ('3', '2'), ('3', '3')],
    #     show_graph,
    #     10,
    #     path="linear3",
    #     name="Linear, two self-loop connected to one"
    # )
    #
    # print_from_list(
    #     [('1', '1'), ('1', '2'), ('2', '2'), ('2', '3'), ('3', '3')],
    #     show_graph,
    #     10,
    #     path="quad1",
    #     name="Quadratic, a chain of 3 self-loops"
    # )
    #
    # print_from_list(
    #     [('1', '1'), ('1', '2'), ('2', '2'), ('2', '3'), ('3', '3'), ('3', '4'), ('4', '4')],
    #     show_graph,
    #     10,
    #     path="cubic1",
    #     name="Cubic, a chain of 4 self-loops."
    # )
    #
    # print_from_list(
    #     [('1', '1'), ('1', '2'), ('2', '4'), ('4', '5'), ('5', '2'), ('2', '3'), ('3', '3')],
    #     show_graph,
    #     10,
    #     path="quad2",
    #     name="Quadratic, a chain of a self-loop, a 3-cycle, and then another self-loop."
    # )
    #
    # print_from_list(
    #     [('1', '1'), ('1', '2'), ('2', '2'), ('2', '4'), ('4', '4'), ('2', '3'), ('3', '3')],
    #     show_graph,
    #     10,
    #     path="quad3",
    #     name="Quadratic, a height 3 tree of self-loops."
    # )
    #
    # print_from_list(
    #     [('1', '2'), ('2', '3'), ('3', '4')],
    #     show_graph,
    #     4,
    #     path="tree1",
    #     name="A simple path that will eventually vanish."
    # )
    #
    # print_from_list(
    #     [('1', '2'), ('2', '3'), ('2', '4'), ('4', '5'), ('4', '6'), ('6', '7')],
    #     show_graph,
    #     6,
    #     path="tree2",
    #     name="A tree that will eventually vanish."
    # )
    #
    # print_from_list(
    #     [('1', '2'), ('2', '3'), ('1', '3')],
    #     show_graph,
    #     4,
    #     path="tree3",
    #     name="Although this looks like a cycle, but it does not contain an oriented cycle. This will vanish."
    # )
    #
    # print_from_list(
    #     [('1', '2'), ('2', '3'), ('3', '4'), ('4', '1')],
    #     show_graph,
    #     5,
    #     path="cycle1",
    #     name="A 4-cycle."
    # )
    #
    # print_from_list(
    #     [('1', '2'), ('2', '1'), ('2', '3'), ('3', '4')],
    #     show_graph,
    #     5,
    #     path="cycle2",
    #     name="A cycle with attached branch."
    # )

    print_from_list(
        [('1', '2'), ('3', '1'), ('2', '3'), ('3', '4'), ('4', '5'), ('6', '4')],
        show_graph,
        7,
        path="cycle3",
        name="A cycle with attached tree."
    )

    # print_from_list(
    #     [('1', '1')],
    #     show_graph,
    #     5,
    #     path="cycle4",
    #     name="A self-loop."
    # )
