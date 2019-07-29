from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.util_hooks import IntervalHook
from .log_hook import LogHook

import numpy as np


class TravelingTrainer(PyHookedModelIterator):
    """The Trainer swaps nodes on the way of the Traveling Salesman.
    If the new path has a lower energy than the old one it is
    accepted, otherwise it is only accepted with a certainty defined
    by the difference between new and old energy.
    """

    def __init__(self, config, root, model, **kwargs):

        self.config = config
        self.root = root
        self.model = model

        loghook = LogHook(model, root)
        modifier = lambda i: i
        log_step = config.get("num_cities", 20) * 100
        ihook = IntervalHook([loghook], log_step, modifier=modifier)

        self.prng = np.random.RandomState(config.get("random_seed", 42))

        self.energy = float("inf")
        super().__init__(config, root, model, hooks=[ihook], hook_freq=1)

    def step_ops(self):
        """Here we define all ops, which are run at every update step.
        """

        return self.train_op

    def train_op(self, model, temperature, swaps, **kwargs):
        """Applies the annealing algorithm to the traveling salesman
        model.

        Args:
            model (Callable): The traveling salesman model, which
                knows about the path, cities and distances on the
                map
            temperature (float): The temperature for the annealing
                process.
            swaps (int, int): Two ints defining the cities to swap
                on the path of the traveling salesman.

        Returns:
            float: The current energy in the system.
        """

        swaps = swaps[0]  # squeeze batch dimension
        idx1, idx2 = swaps
        nodes = np.copy(model.nodes)

        nodes[idx2], nodes[idx1] = nodes[idx1], nodes[idx2]
        path_length = model.get_path_length(nodes)

        new_energy = self.calc_energy(path_length)

        if self.accept(new_energy, self.energy, temperature):
            self.energy = new_energy
            self.model.set_new_path(nodes)

        return self.energy

    def calc_energy(self, path_length):
        return path_length

    def accept(self, new_energy, old_energy, temp):
        if new_energy < old_energy:
            return True
        else:
            probability = np.exp(-(new_energy - old_energy) / temp)

            return np.random.uniform(0, 1) < probability
