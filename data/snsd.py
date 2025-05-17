import torch
import numpy as np
import os
import random
import soundfile as sf

class SNSD:
    
    def __init__(self, root_path:str):
        """Initialize the MS-SNSD dataset support.
        
        Args:
            root_path (str): root path of MS-SNSD dataset
        """
        
        self.categories = ['Babble', 'Neighbor', 'AirConditioner']
        
        self.files = {
            'Babble': [], 'Neighbor': [], 'AirConditioner': []
        }
        
        self.num_noise = {
            'Babble': (1, 1), 'Neighbor': (1, 1), 'AirConditioner': (1, 1)
        }
        
        self.snr_min = 0
        self.snr_max = 20
        
        for dir_path, _, files in os.walk(root_path):
            category = None
            for c in self.categories:
                if c in dir_path:
                    category = c
                    break
            if category is not None:
                for file in files:
                    if '.wav' in file:
                        self.files[category].append(os.path.join(dir_path, file))                
        
    def __call__(self, x, length, category=None, snr=None):
        """Apply MS-SNSD noise to the audio segment.
        
        Args:
            x (any): waveform of utterance
            length : length of noise
            category (str, optional): category of noise ('Babble', 'Neighbor', 'AirConditioner'). Defaults to None for random.
            snr (_type_, optional): Signal-to-noise ratio. Defaults to None for random.
            
        Returns:
            tuple: (noisy_audio, speech_start_idx, speech_end_idx)
        """
        
        if type(x) == torch.Tensor:
            x = x.numpy()
        
        x_size = x.shape[-1]
        x_dB = self.calculate_decibel(x)
        
        # ----------------------- None -> random ----------------------- #
        if category is None:
            category = random.choice(self.categories)
        
        if snr is None:
            snr = random.uniform(self.snr_min, self.snr_max)
        
        # ----------------------- select noises ----------------------- #
        max_samples = min(len(self.files[category]), self.num_noise[category][1])
        min_samples = min(self.num_noise[category][0], max_samples)
        
        noise_files = random.sample(
            self.files[category],
            random.randint(min_samples, max_samples)
        )
        
        noises = []
        for noise in noise_files:
            
            noise, _ = sf.read(noise)
            # random crop
            noise_size = noise.shape[0]
            if noise_size < length:
                shortage = length - noise_size + 1
                noise = np.pad(
                    noise, (0, shortage), 'wrap'
                )
                noise_size = noise.shape[0]
            
            index = random.randint(0, noise_size - length)
            noises.append(noise[index:index + length])

        # ----------------------- inject noise ----------------------- #
        idx = 0
        if len(noises) != 0:
            noise = np.mean(noises, axis=0)
            # calculate dB
            noise_dB = self.calculate_decibel(noise)
            # append
            p = (x_dB - noise_dB - snr)
    
            noise = np.sqrt(10 ** (p / 10)) * noise
            
            idx = random.randint(0, length - x_size)
            noise[idx : idx + x_size] = noise[idx : idx + x_size] + x
            
            x = noise

        return x, idx, idx + x_size-1
        
    
    def calculate_decibel(self, x:torch.Tensor):
        """Calculate the decibel level of a signal
        
        Args:
            x (torch.Tensor): Input signal
            
        Returns:
            float: Decibel level
        """
        assert 0 <= np.mean(x ** 2) + 1e-4
        return 10 * np.log10(np.mean(x ** 2) + 1e-4)