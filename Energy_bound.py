#
from queue import Queue
from typing import List, Any
import graph_generation


class Hexagon:
    def __init__(self, level_in: list, angles_in: int, key_in: object) -> object:
        self.level = level_in
        self.angles = angles_in
        self.key = key_in
        self.parents = [] #keys
        self.children = [] #keys
        self.energy = 0
        self.mountain_valley = ''

    def add_parent(self, pa_key):
        self.parents.append(pa_key)

    def add_child(self, child_key):
        list = self.children + child_key

    def L2_energy(self):
        a = max(self.angles, key=abs)
        self.energy = sum((i * i) / (a * a) for i in self.angles)
        return self.energy

    def children_gen(self):
        global hex_idx
        global hex_dict
        global level_in
        children = []  # new children
        for i in range(6):
            if self.angles[i] > 0:
                c = self.angles.copy()
                i_l = (i - 1) % 6
                i_r = (i + 1) % 6
                c[i_l] = c[i_l] + 2 * c[i]
                c[i_r] = c[i_r] + 2 * c[i]
                c[i] = -c[i]
                k = [key for key, val in hex_dict.items() if
                     set(val.angles) == set(c)]
                self.add_child(k)
                for child_k in self.children:
                    hex_dict[child_k].add_parent(self.key)

                # create a new one
                if len(k) == 0:
                    hex_idx = hex_idx + 1
                    h = Hexagon(self.level + 1, c, hex_idx)
                    if self.level + 1 > level_in:
                        level_in = level_in + 1
                        hex_tree[level_in] = []
                    h.add_parent(self.key)
                    hex_dict[hex_idx] = h
                    children.append(h)

        return children
if __name__ == '__main__':
    hex_idx = 1
    level_in = 1
    hex_dict = {}
    hex_tree = {}
    start_point = Hexagon(level_in, [0, 1, 0, 0, 1, 0], hex_idx)
    hex_dict[hex_idx] = start_point
    hex_tree[level_in] = []
    q = Queue()
    q.put(start_point)
    for i in range(10):
        t = q.get()
        children = t.children_gen()
        for c in children:
            q.put(c)

    # calculate L_2
    energy_list = []
    min_energy = 7
    min_idx = 1
    for key, value in hex_dict.items():
        energy_list.append(value.L2_energy())
        energy = value.L2_energy()
        if min_energy > energy:
            min_energy = energy
            min_idx = key
    #print (energy_list)
    print(min(energy_list))
    print(hex_dict[min_idx].angles)
    #print(hex_dict)