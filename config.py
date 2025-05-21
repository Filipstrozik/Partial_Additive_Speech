import torch

class ExpConfig:
    """
    
    """
    def __init__(self):
        
        self.max_epoch              =  20
        self.batch_size             =  64
        # ------------------ mel spectrogram config ------------------ #
        self.n_mels                 =   80  
        self.sample_rate            =   16000
        self.n_fft                  =   1024
        self.win_length             =   400
        self.hop_length             =   160
        self.window_fn              =   torch.hamming_window
        # ------------------ length of utterances ------------------ #
        self.windows                =   320
        self.train_sample           =   int(self.sample_rate * 3.2) - 1
        self.test_sample            =   int(self.sample_rate * 3.2) - 1
        # ------------------ pre emphasis coefficient ------------- #
        self.pre_emphasis           =   0.97
        # ------------------ model output config ------------------ #
        self.embedding_size         =   192
        self.output_size            =   1211
        # ------------------ loss config ------------------ #
        self.loss_scale             =   15.
        self.loss_margin            =   0.3
        # ------------------ optimizer setting ------------------ #
        self.lr                     =   1e-4
        self.lr_min                 =   1e-7
        self.weight_decay           =   1e-5
        self.amsgrad                =   True
        # ------------------ learning rate scheduler ------------------ #
        self.lr_sch_step_size       =   1       
        self.lr_sch_gamma           =   0.94
        # ------------------ reduce on plateau ------------------ #
        self.lr_sch_patience       =   3
        self.lr_sch_factor          =   0.5

        # ------------------ data augmentation setting ------------------ #
        self.use_pas                =   True # if False: use TAN
        self.pas_min_utter          =   1 * self.sample_rate # duration * sample rate
        # ------------------- exp name ------------------ #

        self.exp_name               =   "ecapa_tdnn_ft_frozen"

class SysConfig:

    def __init__(self):
        # ------------------ path of voxceleb1 ------------------ #
        self.path_vox1_dev_root = "../../datasets/vox1/vox1_dev_wav/wav"
        # self.path_vox1_dev_root = "/data/voxceleb1/train"
        # self.path_vox1_dev_label    = 'label/check_vox1_dev.csv'
        self.path_vox1_dev_label    = 'label/vox1_dev.csv'
        self.path_vox1_test_root = "../../datasets/vox1/vox1_test_wav/wav"
        # self.path_vox1_enroll_label = 'label/check_vox1_enroll.csv'
        self.path_vox1_enroll_label = 'label/vox1_enroll.csv'
        # self.path_vox1_test_label   = 'label/check_vox1_test.csv'
        self.path_vox1_test_label   = 'label/vox1_test.csv'

        self.path_vox1_gaussian_root = (
            "../SV-eval/data/noisy/gaussian/vox1_test_segments_snr_10_noisy_gaussian"
        )
        self.path_vox1_babble_root = "../SV-eval/data/noisy_bg/vox1_test_wav_bq_noise/Babble/vox1_test_wav_snr_10_Babble"

        self.path_vox1_gaussian_csv = "label/gaussian_trials.csv"
        # # ------------------ path of musan ------------------ #
        # self.path_musan_train = "../../datasets/musan_split/train"
        # self.path_musan_test = "../../datasets/musan_split/test"
        # # ------------------ paths of (vox1 test + musan) ------------------ #
        # # key = 'category_snr', value = path of root folder of test
        # self.path_noisy_tests = {
        #     "music_0": "../../datasets/musan_noise_test/music_0",
        #     "music_5": "../../datasets/musan_noise_test/music_5",
        #     "music_10": "../../datasets/musan_noise_test/music_10",
        #     "music_15": "../../datasets/musan_noise_test/music_15",
        #     "music_20": "../../datasets/musan_noise_test/music_20",
        #     "noise_0": "../../datasets/musan_noise_test/noise_0",
        #     "noise_5": "../../datasets/musan_noise_test/noise_5",
        #     "noise_10": "../../datasets/musan_noise_test/noise_10",
        #     "noise_15": "../../datasets/musan_noise_test/noise_15",
        #     "noise_20": "../../datasets/musan_noise_test/noise_20",
        #     "speech_0": "../../datasets/musan_noise_test/speech_0",
        #     "speech_5": "../../datasets/musan_noise_test/speech_5",
        #     "speech_10": "../../datasets/musan_noise_test/speech_10",
        #     "speech_15": "../../datasets/musan_noise_test/speech_15",
        #     "speech_20": "../../datasets/musan_noise_test/speech_20",
        # }
        # ------------------ path of MS-SNSD ------------------ #
        self.path_snsd_train = "../../datasets/SNSD/noise_train"
        self.path_snsd_test = "../../datasets/SNSD/noise_test"

        # ------------------ wandb setting ------------------ #
        self.wandb_disabled         = False
        self.wandb_key = "7f5d1d17417e563f04b7264553f0b853133a0265"
        self.wandb_project          = 'data_pas_aug_finetune'
        self.wandb_entity           = 'ecapa_tdnn_finetune'
        self.wandb_name             = ''
        # ------------------ device setting ------------------ #
        self.num_workers            = 1
        self.device                 =   'cuda:0'
        """device to use for training and testing"""

        self.random_seed            = 1234

if __name__ == "__main__":
    exp = ExpConfig()
    print(exp.n_mels)
