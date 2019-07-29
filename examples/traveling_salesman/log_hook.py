from edflow.hooks.hook import Hook

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import os
import numpy as np


class LogHook(Hook):
    def __init__(self, model, root):
        """Args:
            model (object): Our Traveling Salesman.
            root (str): Path, where the outputs are going to be saved.
        """

        self.model = model
        self.root = root

    def before_step(self, step, fetches, feeds, batch):
        self.current_temperature = feeds["temperature"][0]  # squeeze batch

    def after_step(self, step, last_results):
        """
        Args:
            step (int): The current training step.
            last_results (list): Results from last time this hook was called.
        """

        nodes = self.model.nodes
        city_locs = self.model.city_locations

        pl = "{:.3f}".format(self.model.get_path_length(nodes))
        t = "{:.3f}".format(self.current_temperature)

        annot = {"path length": pl, "temperature": t}

        make_map_plot(nodes, city_locs, step, self.root, annot)


def make_map_plot(nodes, locations, step=None, save_root=".", annot={}):
    fig, ax = plt.subplots(1, 1)

    ordered_cs = locations[nodes]

    segments = np.stack([ordered_cs[1:], ordered_cs[:-1]], axis=0)
    segments = np.concatenate([segments, [[ordered_cs[-1]], [ordered_cs[0]]]], axis=1)
    segments = segments.transpose(1, 0, 2)

    lc = LineCollection(
        segments, cmap=plt.get_cmap("viridis"), norm=plt.Normalize(0, len(nodes))
    )
    lc.set_array(np.arange(0, len(nodes)))
    lc.set_linewidth(3)

    ax.add_collection(lc)

    ax.scatter(locations[:, 0], locations[:, 1])

    title_str = ",".join("{}: {}".format(k, v) for k, v in annot.items())
    ax.set_title(title_str)

    if step is None:
        savename = "salesmap.png"
    else:
        savename = "salesmap-{}.png".format(step)
    fig.savefig(os.path.join(save_root, savename))


if __name__ == "__main__":
    from salesman import TravelingSalesman

    TS = TravelingSalesman({"num_cities": 20})

    make_map_plot(TS.nodes, TS.city_locations)
