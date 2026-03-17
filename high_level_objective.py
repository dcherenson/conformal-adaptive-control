import numpy as np

class HighLevelObjective:
    """
    Provides the goal trajectory or reference.
    """
    def __init__(self):
        pass

    def get_reference(self, t: float) -> np.ndarray:
        """The objective is to regulate the state to [p_target_x, p_target_y, v_target_x, v_target_y]."""
        return np.array([2.0, 2.0, 0.0, 0.0])
