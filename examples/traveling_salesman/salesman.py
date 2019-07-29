import numpy as np


class TravelingSalesman(object):
    """The Traveling Salesman knows about all cities he or she needs to
    visit and how far they are apart. He or she also keeps track of
    the path to visit all cities which is currently planned.
    """

    def __init__(self, config):
        """The config can be used to specify the following things for the
        Salesman:
            num_cities (int): The number of randomly genreated cities.
            random_seed (int): An optional random seed for reproducability.
        """

        self.num_cities = config.get("num_cities", 20)
        self.prng = np.random.RandomState(config.get("random_seed", 42))

        loc_shape = [self.num_cities, 2]
        self.city_locations = locs = self.prng.uniform(0, 100, size=loc_shape)

        self.distance_matrix = np.linalg.norm(
            locs[None, :, :] - locs[:, None, :], axis=-1
        )

        self.nodes = np.arange(self.num_cities)

    def swap_nodes(self, idx1, idx2):
        """Returns the nodes without applying the swap."""

        n1 = self.nodes[idx1]
        n2 = self.nodes[idx2]

        nodes = np.copy(self.nodes)
        nodes[idx2], nodes[idx1] = nodes[idx1], nodes[idx2]

        return nodes

    def get_path_length(self, nodes):
        length = 0
        for from_node, to_node in zip(nodes[1:], nodes[:-1]):
            length += self.distance_matrix[from_node][to_node]

        # And close the loop!
        length += self.distance_matrix[nodes[-1]][nodes[0]]

        return length

    def set_new_path(self, nodes):
        self.nodes = nodes
