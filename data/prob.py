import torch
import numpy as np
import random


class ProbabilisticNoiseSNR:
    def __init__(self):
        """Initialize the Probabilistic Noise generator with SNR control."""
        self.noise_types = ["gaussian", "rayleigh", "poisson"]
        self.snr_min = 0
        self.snr_max = 20

    def calculate_decibel(self, x):
        """Calculate the decibel level of a signal."""
        mean_square = np.mean(x**2)
        mean_square = max(mean_square, 0) + 1e-4  # avoid log(0)
        return 10 * np.log10(mean_square)

    def __call__(self, x, length=None, noise_type=None, snr=None):
        """
        Apply probabilistic noise to the audio segment with SNR control.
        The signal is injected at a random position into a longer noise sample.
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        x_size = x.shape[-1]
        if length is None:
            length = x_size

        if noise_type is None:
            noise_type = random.choice(self.noise_types)

        if snr is None:
            snr = random.uniform(self.snr_min, self.snr_max)

        # Generate base noise of full length
        full_noise = np.random.uniform(-1, 1, length)

        # Randomly choose insertion index
        idx = random.randint(0, length - x_size) if length > x_size else 0

        # Extract segment of noise where signal will be placed
        noise_segment = full_noise[idx : idx + x_size]

        # Scale noise segment to match desired SNR
        scaled_noise_segment = add_scaled_noise_snr_segment(
            x, noise_segment, target_snr=snr, noise_type=noise_type
        )

        # Inject signal+scaled_noise into the full noise array
        full_noise[idx : idx + x_size] = scaled_noise_segment

        return full_noise, idx, idx + x_size - 1


def add_scaled_noise_snr_segment(
    signal: np.ndarray, noise: np.ndarray, target_snr: float, noise_type="gaussian"
) -> np.ndarray:
    """
    Generate noise of specified type and scale it to achieve the target SNR
    when added to the provided signal.
    """
    rms_signal = np.sqrt(np.mean(signal**2))
    snr_linear = 10 ** (target_snr / 20)
    rms_noise = rms_signal / snr_linear

    match noise_type:
        case "gaussian":
            noise = np.random.normal(0, rms_noise, signal.shape)
        case "rayleigh":
            sigma = rms_noise / np.sqrt(2 - np.pi / 2)
            noise = np.random.rayleigh(sigma, signal.shape)
            noise -= np.mean(noise)
            noise *= rms_noise / np.sqrt(np.mean(noise**2))
            noise *= np.sign(signal)
        case "poisson":
            lambda_vals = np.abs(signal) * rms_noise
            noise = np.random.poisson(lambda_vals, signal.shape) - lambda_vals
            noise -= np.mean(noise)
            noise *= rms_noise / np.sqrt(np.mean(noise**2))
            noise *= np.sign(signal)
        case _:
            raise ValueError("Unsupported noise type.")

    return signal + noise
