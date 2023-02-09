from typing import Union
import numpy as np


def get_statistical_info(values: Union[np.ndarray, list]):
    if isinstance(values, list):
        values = np.asarray(values)

    result = {
        'mean': np.mean(values),
        'worst': np.max(values),
        'best': np.min(values),
        'std': np.std(values),
        'median': np.median(values),
    }
    return result
