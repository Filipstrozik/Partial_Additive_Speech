import torch
import numpy as np
import random

class ProbabilisticNoiseSNR:
    def __init__(self):
        """Initialize the Probabilistic Noise generator with SNR control."""
        self.noise_types = ['gaussian', 'rayleigh', 'poisson']
        self.snr_min = 0
        self.snr_max = 20

    def calculate_decibel(self, x):
        """Calculate the decibel level of a signal.
        
        Args:
            x (np.ndarray): Input signal
            
        Returns:
            float: Decibel level
        """
        assert 0 <= np.mean(x ** 2) + 1e-4
        return 10 * np.log10(np.mean(x ** 2) + 1e-4)

    def __call__(self, x, length=None, noise_type=None, snr=None):
        """Apply probabilistic noise to the audio segment with SNR control.
        
        Args:
            x (any): waveform of utterance
            length (int, optional): length of noise. If None, uses length of x
            noise_type (str, optional): type of noise. Defaults to None for random.
            snr (float, optional): Signal-to-noise ratio. Defaults to None for random.
            
        Returns:
            tuple: (noisy_audio, start_idx, end_idx)
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        x_size = x.shape[-1]
        if length is None:
            length = x_size

        # Calculate input signal dB
        x_dB = self.calculate_decibel(x)

        # Random selection if not specified
        if noise_type is None:
            noise_type = random.choice(self.noise_types)

        if snr is None:
            snr = random.uniform(self.snr_min, self.snr_max)

        # Generate base noise
        noise = add_scaled_noise_snr(np.zeros(length), snr, noise_type=noise_type)
        
        # Calculate noise dB
        noise_dB = self.calculate_decibel(noise)
        
        # SNSD-style injection
        p = (x_dB - noise_dB - snr)
        noise = np.sqrt(10 ** (p / 10)) * noise
        
        # Random positioning of signal within noise
        idx = random.randint(0, length - x_size) if length > x_size else 0
        noise[idx:idx + x_size] = noise[idx:idx + x_size] + x

        return noise, idx, idx + x_size - 1


def add_scaled_noise_snr(
    signal: np.ndarray, target_snr: float, noise_type="gaussian", log_metrics=False
) -> np.ndarray:
    rms_signal = np.sqrt(np.mean(signal**2))
    snr_linear = 10 ** (target_snr / 20)
    rms_noise = rms_signal / snr_linear

    match noise_type:
        case "gaussian":
            noise = np.random.normal(0, rms_noise, signal.shape)
        case "rayleigh":
            sigma = rms_noise / np.sqrt(2 - np.pi / 2)
            noise = np.random.rayleigh(sigma, signal.shape)
            noise = noise - np.mean(noise)
            signs = np.sign(signal)
            noise = noise * rms_noise / np.sqrt(np.mean(noise**2))
            noise = noise * signs
        case "poisson":
            lambda_vals = np.abs(signal) * rms_noise
            noise = np.random.poisson(lambda_vals, signal.shape) - lambda_vals
            noise = noise - np.mean(noise)
            signs = np.sign(signal)
            noise = noise * rms_noise / np.sqrt(np.mean(noise**2))
            noise = noise * signs
        case _:
            raise ValueError("Unsupported noise type.")

    if log_metrics:
        rms_actual_noise = np.sqrt(np.mean(noise**2))
        snr = (
            20 * np.log10(rms_signal / rms_actual_noise)
            if rms_actual_noise > 0
            else float("inf")
        )

    return signal + noise
