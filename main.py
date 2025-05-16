import wandb
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import config
import os
from models.se_resnet import SEResNet34
from models.ecapa_tdnn import ECAPA_TDNN
from trainer.exp_trainer import Trainer
import wespeaker

class Main:

    def __init__(self):

        sys_config = config.SysConfig()
        exp_config = config.ExpConfig()

        ###
        ###    seed setting
        ###
        seed = sys_config.random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        #  cudnn setting to enhance speed of dilated conv
        cudnn.deterministic = False
        cudnn.benchmark = True

        # cudnn.deterministic = True
        # cudnn.benchmark = False

        ###
        ###    wandb setting
        ###
        if sys_config.wandb_disabled:
            os.system("wandb disabled")

        os.system(f"wandb login {sys_config.wandb_key}")
        wandb.init(
            project = sys_config.wandb_project,
            entity  = sys_config.wandb_entity,
            name    = sys_config.wandb_name
        )

        ###
        ###    training environment setting
        ###
        self.max_epoch = exp_config.max_epoch
        # self.model = SEResNet34().to(sys_config.device)
        # self.model = ECAPA_TDNN().to(sys_config.device)
        # print(self.model)
        ecapa_tdnn = wespeaker.load_model_local("models/voxceleb_ECAPA1024")
        self.model = ecapa_tdnn.model.to(sys_config.device)
        # self.model = wespeaker_model.model.to(sys_config.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = exp_config.lr,
            weight_decay = exp_config.weight_decay,
            amsgrad=exp_config.amsgrad
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=exp_config.lr_sch_step_size,
            gamma=exp_config.lr_sch_gamma
        )

        self.trainer = Trainer(model=self.model, optimizer=self.optimizer)

    def start(self):

        min_eer = None
        model_state = {}

        for epoch in range(1, self.max_epoch + 1):

            # --------------- train --------------- #
            self.trainer.train()

            self.lr_scheduler.step()  

            if epoch < 1:
                continue

            # Run testing more frequently
            test_frequency = 5  # Test every 5 epochs
            save_frequency = 1  # Save checkpoints every epoch

            if epoch % test_frequency == 0:
                # --------------- test --------------- #
                eer = self.trainer.test(epoch)

                # --------------- eer check and save --------------- #
                if min_eer is None:
                    min_eer = eer
                    for test_type in min_eer.keys():
                        model_state[test_type] = self.model.state_dict()
                else:
                    for test_type in min_eer.keys():
                        # Always save the latest model state
                        latest_model_path = f"checkpoints/latest_{test_type}.pth"
                        os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
                        torch.save(self.model.state_dict(), latest_model_path)
                        wandb.save(latest_model_path)
                        
                        # save process by test type - only if better than previous best
                        if min_eer[test_type] < eer[test_type]:
                            continue

                        min_eer[test_type] = eer[test_type]
                        model_state[test_type] = self.model.state_dict()

                        # Save best model
                        best_file_name = f"checkpoints/{epoch}_{test_type}_{min_eer[test_type]:.4f}_best.pth"
                        os.makedirs(os.path.dirname(best_file_name), exist_ok=True)
                        torch.save(model_state[test_type], best_file_name)
                        wandb.save(best_file_name)
                
                # Additionally save periodic checkpoints regardless of performance
                if epoch % save_frequency == 0:
                    periodic_checkpoint = f"checkpoints/epoch_{epoch}.pth"
                    os.makedirs(os.path.dirname(periodic_checkpoint), exist_ok=True)
                    torch.save(self.model.state_dict(), periodic_checkpoint)
                    wandb.save(periodic_checkpoint)

                # --------------- log and schedule learning rate --------------- #
                print(f"epoch: {epoch} \neer: {eer} \nmin_eer:{min_eer}")        


if __name__ == '__main__':
    program = Main()
    program.start()
