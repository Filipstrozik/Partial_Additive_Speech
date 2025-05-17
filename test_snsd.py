import torch
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
from data.snsd import SNSD

def test_snsd():
    """
    Test the SNSD class by applying different noise types to an audio sample
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

    # Initialize SNSD with dataset path (update path)
    snsd_path = "../../datasets/SNSD/noise_train"  # Update this path
    snsd = SNSD(root_path=snsd_path)
    print(snsd.files)

    # Apply each noise type
    for category in snsd.categories:
        # Apply noise with random SNR
        noisy_audio, start_idx, end_idx = snsd(audio_tensor, len(audio), category=category, snr=0)

        # Save the noisy audio
        output_file = f"output/noisy_{category.lower()}.wav"
        sf.write(output_file, noisy_audio, sr)

        print(f"Created {category} noisy audio: {output_file}")
        print(f"  - Speech segment: {start_idx} to {end_idx}")

        # Plot waveform for visualization
        plt.figure(figsize=(10, 4))
        plt.plot(noisy_audio)
        plt.axvline(x=start_idx, color='r', linestyle='--', label='Speech start')
        plt.axvline(x=end_idx, color='g', linestyle='--', label='Speech end')
        plt.title(f"Noisy Audio with {category} Noise (SNR=10dB)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(f"output/waveform_{category.lower()}.png")
        plt.close()

if __name__ == "__main__":
    test_snsd()
