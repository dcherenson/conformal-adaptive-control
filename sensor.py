import numpy as np

class Sensor:
    """
    Represents the measurement subsystem.
    """
    def __init__(self, C: np.ndarray, noise_std: float = 0.1):
        self.C = C
        self.noise_std = noise_std

    def measure(self, x: np.ndarray) -> np.ndarray:
        """Returns noisy measurement y = Cx + v"""
        noise = np.random.normal(0, self.noise_std, size=self.C.shape[0])
        return self.C @ x + noise
