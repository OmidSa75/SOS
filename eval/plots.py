from typing import Union

import numpy as np
from matplotlib import pyplot as plt


def plot_values(steps: int, values: Union[list, np.ndarray], title_name: str):
    plt.figure(figsize=(10, 7))
    plt.plot(range(steps), values)
    plt.title(title_name)
    plt.grid()
    plt.show()


def plot_boxplot(values: Union[list, np.ndarray], title_name: str):
    plt.figure(figsize=(10, 7))
    plt.boxplot(values)
    plt.title(title_name)
    plt.grid()
    plt.show()
