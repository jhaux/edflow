"""
The Dataset supplies us with the values we need at every trainings
step. In our case this is the annealing temperature and which cities
to swap on the path of the Traveling Salesman.
"""


from edflow.data.dataset import DatasetMixin
import numpy as np


# This is a model!
class SalesData(DatasetMixin):
    """The Atlas contains all cities and a distance matrix with their
    distances.
    """

    def __init__(self, config):
        """Config can contain:
            start_temperature (float): The starting temperature of the
                annealing process. Defaults to 10000.
            end_temperature (float): The end temperature of the annealing
                process. Defaults to 0.
            num_steps (int): How many anneal steps to take. Defaults
                to 1e6.
            anneal_type (str): One of ``[lin, exp, sqrt, log]``. The
                way how the temperature is decreased. Defaults to ``lin``.
            num_cities (int): How many cities the traveling salesman can
                visit. Defaults to 20.
            random_seed (int): Passed to ``numpy.random.RandomState``. Defaults
                to 42 obviously.
            successive_swap (bool): If True, only successive cities are being
                swapped, otherwise swapping is random. Defaults to True.
        """

        self.t0 = self.temp = config.get("start_temperature", 10000)
        self.tf = config.get("end_temperature", 0)
        self.n_steps = config.get("num_steps", 1e6)

        self.anneal_type = config.get("anneal_type", "lin")
        assert self.anneal_type in ["lin", "exp", "sqrt", "log"], (
            "anneal_type must be one of `lin, exp, sqrt, log` but "
            "is {}".format(self.anneal_type)
        )

        self.n_cities = config.get("num_cities", 20)
        self.prng = np.random.RandomState(config.get("random_seed", 42))

        self.succesive = config.get("successive_swap", True)

    def get_example(self, idx):
        """Gives the current temperature and the swap indices.

        Example output:

        """

        if self.anneal_type == "lin":
            step = idx // (self.n_cities * 100)
            self.temp = linear_decay(step, self.t0, self.tf, self.n_steps)
            self.temp = np.clip(self.temp, self.t0, self.tf)
        else:
            if idx % (self.n_cities * 100) == 0:
                self.temp = 0.99 * self.temp

        city_idx1 = self.prng.randint(self.n_cities)
        if self.succesive:
            city_idx2 = (city_idx1 + 1) % (self.n_cities)
        else:
            city_idx2 = self.prng.randint(self.num_cities)

        out = {"temperature": self.temp, "swaps": [city_idx1, city_idx2]}

        return out

    def __len__(self):
        return self.n_steps * self.n_cities * 100


def linear_decay(current, start, stop, total):
    return (start - stop) * current / total + start


def exp_decay(current, start, stop, total):
    return (stop - start) * current / total + start


def log_decay(current, start, stop, total):
    return (stop - start) * current / total + start


def sqrt_decay(current, start, stop, total):
    return (stop - start) * current / total + start
