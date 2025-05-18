import torch
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
from data.prob import ProbabilisticNoiseSNR

def test_prob():
    """
    Test the ProbabilisticNoiseSNR class by applying different noise types to an audio sample
    and saving the results.
    """
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Load sample audio (replace with your audio path)
    sample_path = (
        "../../datasets/vox1/vox1_dev_wav/wav/id10001/1zcIwhmdeo4/00001.wav"  # Update this path
    )
    if not os.path.exists(sample_path):
        print(f"Sample audio file not found: {sample_path}")
        print("Please update the sample_path variable.")
        return

    audio, sr = sf.read(sample_path)

    # Convert to torch tensor if needed
    audio_tensor = torch.tensor(audio)

    # Initialize ProbabilisticNoiseSNR
    prob_noise = ProbabilisticNoiseSNR()

    # Apply each noise type
    for noise_type in prob_noise.noise_types:
        # Apply noise with fixed SNR for demonstration
        noisy_audio, start_idx, end_idx = prob_noise(audio_tensor, len(audio), noise_type=noise_type, snr=20)

        # Save the noisy audio
        output_file = f"output/noisy_{noise_type.lower()}.wav"
        sf.write(output_file, noisy_audio, sr)

        print(f"Created {noise_type} noisy audio: {output_file}")
        print(f"  - Speech segment: {start_idx} to {end_idx}")

        # Plot waveform for visualization
        plt.figure(figsize=(10, 4))
        plt.plot(noisy_audio)
        plt.axvline(x=start_idx, color='r', linestyle='--', label='Speech start')
        plt.axvline(x=end_idx, color='g', linestyle='--', label='Speech end')
        plt.title(f"Noisy Audio with {noise_type} Noise (SNR=10dB)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(f"output/waveform_{noise_type.lower()}.png")
        plt.close()

if __name__ == "__main__":
    test_prob()