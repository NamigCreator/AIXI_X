from abc import abstractmethod
from typing import Optional
import numpy as np

from .misc import window_image

class Filter:
    @abstractmethod
    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        pass


class FilterBrainPresent(Filter):
    def __init__(self, 
            window_center : int = 40,
            window_width : int = 80,
            threshold_occupancy : float = 0.02,     
            ):
        self.window_center = window_center
        self.window_width = window_width
        self.threshold_occupancy = threshold_occupancy

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        windowed = window_image(image, 
            window_center=self.window_center,
            window_width=self.window_width,
        )
        occupancy = np.mean((windowed > 0) & (windowed < 1))
        if occupancy >= self.threshold_occupancy:
            return image
        else:
            return None
        

def get_filter_by_name(name: str, **kwargs):
    if name == "brain_present":
        return FilterBrainPresent(**kwargs)
    else:
        raise ValueError(f"Unknown filter: {name}")