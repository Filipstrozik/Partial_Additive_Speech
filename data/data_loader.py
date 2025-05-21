import torch.utils.data as data
import torchaudio.compliance.kaldi as kaldi
import torch
import config
import torchaudio
import pandas as pd
import numpy as np
import os
import random
from data.musan import MusanNoise
from data.pas import Pas

from data.snsd import SNSD
from data.prob import ProbabilisticNoiseSNR

class Vox1DevSet(data.Dataset):
    """
    Voxceleb1 development data without data augmentation
    """
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        super(Vox1DevSet, self).__init__()
        
        self.path_root_dir = sys_config.path_vox1_dev_root
        self.utter_length = exp_config.train_sample
        self.anno_table = pd.read_csv(sys_config.path_vox1_dev_label, delim_whitespace = True)
        
    def __len__(self):
        return len(self.anno_table)
    
    def __getitem__(self, idx):
        path = os.path.join(self.path_root_dir, self.anno_table.iloc[idx, 2])
        
        # -------------------- get utterance and speaker number-------------------- #
        utter, _ = torchaudio.load(path)
        utter = torch.squeeze(utter)
        spk_id = int(self.anno_table.iloc[idx, 0]) - 1
        
        # -------------------- resize utterance -------------------- #
        utter_len = len(utter)
        if utter_len < self.utter_length:
            tmp = [utter for i in range(0, (self.utter_length // utter_len))]
            
            residue = self.utter_length % utter_len
            if residue > 0: tmp.append(utter[0:residue])
            
            utter = torch.cat(tmp, dim=0)
        
        # -------------------- select random segments from utterance ------------------------ #
        start_seg = random.randint(0, utter_len - self.utter_length)
        utter = utter[start_seg : start_seg + self.utter_length]        
        
        return utter, spk_id

class Vox1EnrollSet(data.Dataset):
    """
    Vox1EnrollSet is used to prepare embeddings for testing.  \\
    This is used for test in clean scenario. \\ 
    
    `__getitem__(idx)` method returns `(utter, spk_id)`  \\
    
    `spk_id:str`            - path of an utterance file, this can be used as identity for test. Thus, this is used as identity for enrollment utterance.
    `utter:torch.Tensor`    - Batched utterance of a speaker identified as `spk_id`. The shape is like [batch, utterance_length]
    
    """
    
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()) -> None:
        super(Vox1EnrollSet, self).__init__()
        self.path_root_dir = sys_config.path_vox1_test_root
        self.utter_length = exp_config.test_sample
        self.anno_table = pd.read_csv(sys_config.path_vox1_enroll_label, delim_whitespace = True)
        
    def __len__(self):
        return len(self.anno_table)
    
    def __getitem__(self, idx):
        
        # ------------------- speaker's id is a path of the utterance audio file ----------------- #
        spk_id = self.anno_table.iloc[idx, 0]
        
        # -------------------- get utterance-------------------- #
        path = os.path.join(self.path_root_dir, spk_id)
        utter, _ = torchaudio.load(path)
        utter = torch.squeeze(utter)
        
        # -------------------- resize utterance -------------------- #
        utter_len = len(utter)
        if utter_len < self.utter_length:
            tmp = [utter for i in range(0, (self.utter_length // utter_len))]
            
            residue = self.utter_length % utter_len
            if residue > 0: tmp.append(utter[0:residue])
            
            utter = torch.cat(tmp, dim=0)
        
        # -------------------- make batch -------------------------- #
        #seg_idx = np.arange(start=0, stop=len(utter) - self.utter_length, step=self.utter_length, dtype=int)
        seg_idx = torch.linspace(start=0, end=utter_len-self.utter_length, steps=30, dtype=int).tolist()
        tmp = []
        for start in seg_idx:
            tmp.append(utter[start : start + self.utter_length])
        #tmp.append(utter[-self.utter_length:])
        utter = torch.stack(tmp, dim = 0)        
        # shape of utter == [num_seg, self.utter_length]

        # -------------------- apply fbank -------------------- #
        fbank_features = []
        for i in range(utter.shape[0]):
            segment = utter[i].unsqueeze(0)  # Add channel dimension [1, self.utter_length]
            fbank = compute_fbank(segment)   # Calculate FBANK features
            fbank_features.append(fbank)
        
        # Stack all FBANK features along batch dimension
        utter = torch.stack(fbank_features, dim=0)

        return utter, spk_id


class CustomNoisyEnrollSet(data.Dataset):
    """
    Dataset for enrollment with a custom directory structure where audio files
    are organized by speaker ID directly without video name subdirectories.

    The directory structure is expected to be:
    root_path/
        speaker_id1/
            audio1.wav
            audio2.wav
            ...
        speaker_id2/
            audio1.wav
            ...

    `__getitem__(idx)` returns `(utter, spk_id)` where:
    `spk_id:str`            - The speaker ID (directory name)
    `utter:torch.Tensor`    - Batched utterance features of shape [batch, time, features]
    """

    def __init__(
        self,
        root_path,
        audio_extension=".wav",
        sys_config=config.SysConfig(),
        exp_config=config.ExpConfig(),
    ) -> None:
        super(CustomNoisyEnrollSet, self).__init__()
        self.path_root_dir = root_path
        self.utter_length = exp_config.test_sample
        self.audio_extension = audio_extension

        # Build speaker-to-audio mapping
        self.samples = []
        for speaker_id in os.listdir(root_path):
            speaker_dir = os.path.join(root_path, speaker_id)
            if os.path.isdir(speaker_dir):
                for audio_file in os.listdir(speaker_dir):
                    if audio_file.endswith(audio_extension):
                        self.samples.append(
                            {
                                "speaker_id": speaker_id,
                                "audio_path": os.path.join(speaker_id, audio_file),
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # speaker_id = sample["speaker_id"]
        audio_path = sample["audio_path"]

        # Get utterance
        path = os.path.join(self.path_root_dir, audio_path)

        # Load only a portion of the audio for quick debugging
        # Use a smaller segment for faster loading
        utter, sr = torchaudio.load(path)
        utter = torch.squeeze(utter)

        # Apply fbank directly without batching for now
        if utter.dim() == 1:  # If mono
            utter = utter.unsqueeze(0)  # Add channel dimension
        utter = compute_fbank(utter)

        # add 1 more dimension for batch to be [batch, time, features]
        utter = utter.unsqueeze(0)

        return utter, audio_path


def vox1_trial_list(sys_config=config.SysConfig()):
    """_summary_
    Trial list for test.
    
    Returns:
        list: list of pairs. pair: (utterance1 path, utterance2 path, is same).
        The path means relative path of the utterance file.
        This system uses the path as enrollment id.
    """
    anno_table = pd.read_csv(sys_config.path_vox1_test_label, delim_whitespace = True)
    result = []
    for idx in range(0, len(anno_table)):
        is_same = anno_table.iloc[idx, 0]
        spk_id1 = anno_table.iloc[idx, 1]
        spk_id2 = anno_table.iloc[idx, 2]
        result.append([spk_id1, spk_id2, is_same])
    return result


def load_custom_trial_list(csv_path):
    """Load a custom trial list from CSV file for evaluation"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trial list CSV not found at {csv_path}")

    trials = []
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                label = int(parts[0])
                audio1 = parts[1]
                audio2 = parts[2]
                trials.append([audio1, audio2, label])

    return trials


def compute_fbank(
    wavform,
    sample_rate=16000,
    num_mel_bins=80,
    frame_length=25,
    frame_shift=10,
    cmn=True,
):
    feat = kaldi.fbank(
        wavform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        sample_frequency=sample_rate,
    )
    if cmn:
        feat = feat - torch.mean(feat, 0)
    return feat

class PasTrainSet(data.Dataset):
    """_summary_
    Training dataset with PAS data augmentation method.
    For 1/4 chance, do not augmentation.
    """

    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        super(PasTrainSet, self).__init__()
        self.exp_config = exp_config
        self.vox1_dev = Vox1DevSet()
        # self.pas = Pas(root_path=sys_config.path_musan_train)
        self.snsd = SNSD(root_path=sys_config.path_snsd_train)
        self.prob_noise = ProbabilisticNoiseSNR()

    def __len__(self):
        return len(self.vox1_dev)

    def __getitem__(self, idx):
        utter, spk_id = self.vox1_dev.__getitem__(idx)

        if random.randint(0, 3) == 0:
            utter = utter.unsqueeze(0)
            utter = compute_fbank(utter)
            return utter, spk_id

        length = utter.shape[-1]

        utter_len = random.randint(self.exp_config.pas_min_utter, length)
        utter_s = random.randint(0, length - utter_len)
        utter = utter[utter_s : utter_s + utter_len]

        # utter, _, _ = self.pas(x=utter, length=length)

        # with 0.5 probability add SNSD noise, or add regular noise (to be implemented)
        if random.randint(0, 1) == 0:
            utter, _, _ = self.snsd(x=utter, length=length)
        else:
            utter, _, _ = self.prob_noise(x=utter, length=length)

        utter = torch.Tensor(utter)

        ## COMPUTE FBANK INSIDE DATASET ON (C, N)
        utter = utter.unsqueeze(0)
        utter = compute_fbank(utter)

        return utter, spk_id


class OriginalTrainSet(data.Dataset):
    """_summary_
    Original training dataset without data augmentation.
    """

    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        super(OriginalTrainSet, self).__init__()
        self.vox1_dev = Vox1DevSet()

    def __len__(self):
        return len(self.vox1_dev)

    def __getitem__(self, idx):
        utter, spk_id = self.vox1_dev.__getitem__(idx)
        
        # -------------------- apply fbank -------------------- #
        utter = utter.unsqueeze(0)
        utter = compute_fbank(utter)

        return utter, spk_id

class TanTrainSet(data.Dataset):
    """_summary_
    TAN( Traditional Additive Noise ) train set. \\
    This randomly adds Musan noise to Vox1DevSet for entire duration of each speech. \\
    This gives clean(1/4 chance) or noisy(3/4 chance) utterance. \\
    
    The probability is based on the number of categories of MUSAN.(categories are splitted in speech, noise, music)
    """
    
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        super(TanTrainSet, self).__init__()
        self.vox1_dev = Vox1DevSet()
        self.musan = MusanNoise(root_path=sys_config.path_musan_train)
        
    def __len__(self):
        return len(self.vox1_dev)
    
    def __getitem__(self, idx):
        utter, spk_id = self.vox1_dev.__getitem__(idx)
        
        add_noise = random.randint(0, 3)
        
        if add_noise != 0: # noisy train data : 3/4 chance
            utter = torch.Tensor(self.musan(utter)) # MusanNoise randomly select category of noise
        
        # -------------------- apply fbank -------------------- #
        utter = utter.unsqueeze(0)
        utter = compute_fbank(utter)

        return utter, spk_id


class NoisyEnrollSet(data.Dataset):
    """_summary_
    Dataset used for test in noisy environment. \\
    This reads audio files from the root_path parameter of initiator. \\
    """
    
    def __init__(self, root_path, sys_config=config.SysConfig(), exp_config=config.ExpConfig()) -> None:
        super(NoisyEnrollSet, self).__init__()
        self.path_root_dir = root_path
        self.utter_length = exp_config.test_sample
        self.anno_table = pd.read_csv(sys_config.path_vox1_enroll_label, delim_whitespace = True)
        
    def __len__(self):
        return len(self.anno_table)
    
    def __getitem__(self, idx):
        
        # ------------------- speaker's id is a path of the utterance audio file ----------------- #
        spk_id = self.anno_table.iloc[idx, 0]
        
        # -------------------- get utterance-------------------- #
        path = os.path.join(self.path_root_dir, spk_id)
        utter, _ = torchaudio.load(path)
        utter = torch.squeeze(utter)
        
        # -------------------- resize utterance -------------------- #
        utter_len = len(utter)
        if utter_len < self.utter_length:
            tmp = [utter for i in range(0, (self.utter_length // utter_len))]
            
            residue = self.utter_length % utter_len
            if residue > 0: tmp.append(utter[0:residue])
            
            utter = torch.cat(tmp, dim=0)
        
        # -------------------- make batch -------------------------- #
        seg_idx = torch.linspace(start=0, end=utter_len-self.utter_length, steps=30, dtype=int).tolist()
        tmp = []
        for start in seg_idx:
            tmp.append(utter[start : start + self.utter_length])
        
        utter = torch.stack(tmp, dim = 0)        
        # shape of utter == [num_seg, self.utter_length]


        # -------------------- apply fbank -------------------- #
        fbank_features = []
        for i in range(utter.shape[0]):
            segment = utter[i].unsqueeze(0)  # Add channel dimension [1, self.utter_length]
            fbank = compute_fbank(segment)   # Calculate FBANK features
            fbank_features.append(fbank)
        
        # Stack all FBANK features along batch dimension
        utter = torch.stack(fbank_features, dim=0)

        return utter, spk_id
