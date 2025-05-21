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

        self.sys_config = config.SysConfig()
        self.exp_config = config.ExpConfig()

        ###
        ###    seed setting
        ###
        seed = self.sys_config.random_seed
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
        if self.sys_config.wandb_disabled:
            os.system("wandb disabled")

        os.system(f"wandb login {self.sys_config.wandb_key}")
        wandb.init(
            project = self.sys_config.wandb_project,
            entity  = self.sys_config.wandb_entity,
            name    = self.sys_config.wandb_name
        )

        ###
        ###    Create experiment checkpoint directory
        ###
        self.exp_name = self.exp_config.exp_name
        self.checkpoint_dir = os.path.join("checkpoints", self.exp_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Experiment checkpoint directory: {self.checkpoint_dir}")

        ###
        ###    training environment setting
        ###
        self.max_epoch = self.exp_config.max_epoch
        # self.model = SEResNet34().to(self.sys_config.device)
        # self.model = ECAPA_TDNN().to(self.sys_config.device)
        # print(self.model)
        ecapa_tdnn = wespeaker.load_model_local("models/voxceleb_ECAPA1024")
        self.model = ecapa_tdnn.model.to(self.sys_config.device)
        # self.model = wespeaker_model.model.to(self.sys_config.device)

        self.freeze_model()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = self.exp_config.lr,
            weight_decay = self.exp_config.weight_decay,
            amsgrad=self.exp_config.amsgrad
        )

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size=self.exp_config.lr_sch_step_size,
        #     gamma=self.exp_config.lr_sch_gamma
        # )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.exp_config.lr_sch_factor,
            patience=self.exp_config.lr_sch_patience,
        )

        self.trainer = Trainer(model=self.model, optimizer=self.optimizer)

        # Load checkpoint if it exists
        self.start_epoch = 1
        self.min_eer = None
        self.model_state = {}
        self.resume_checkpoint()

    def freeze_model(self):
        for param in self.model.layer1.parameters():
            param.requires_grad = False

        for param in self.model.layer2.parameters():
            param.requires_grad = False

        for param in self.model.layer3.parameters():
            param.requires_grad = False

        # Partially unfreeze layer4 - only make SE_Connect trainable
        for name, param in self.model.layer4.named_parameters():
            if "se_res2block.3" in name:  # SE_Connect module
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Ensure upper layers are trainable (they are by default, but being explicit)
        for param in self.model.conv.parameters():
            param.requires_grad = True

        for param in self.model.pool.parameters():
            param.requires_grad = True

        for param in self.model.bn.parameters():
            param.requires_grad = True

        for param in self.model.linear.parameters():
            param.requires_grad = True

        # Verify which layers are trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})"
        )

    def save_checkpoint(self, epoch, min_eer=None, model_state=None, is_best=False, test_type=None):
        """Save checkpoint with model, optimizer and lr_scheduler states"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'min_eer': min_eer,
            'model_state': model_state,
        }

        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_checkpoint_path)
        wandb.save(latest_checkpoint_path)

        # Save periodic checkpoint
        periodic_checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save(checkpoint, periodic_checkpoint_path)
        wandb.save(periodic_checkpoint_path)

        # Save best model if specified
        if is_best and test_type is not None:
            best_path = os.path.join(self.checkpoint_dir, f"best_{test_type}_{min_eer[test_type]:.4f}.pth")
            torch.save(checkpoint, best_path)
            wandb.save(best_path)

    def resume_checkpoint(self):
        """Resume from latest checkpoint if it exists"""
        latest_checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")

        if os.path.exists(latest_checkpoint_path):
            print(f"Resuming from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.min_eer = checkpoint['min_eer']
            self.model_state = checkpoint['model_state'] if 'model_state' in checkpoint else {}

            print(f"Resumed training from epoch {self.start_epoch}")
            return True

        return False

    def start(self):
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            # --------------- train --------------- #

            eer = self.trainer.test(epoch)
            break

            self.trainer.train()

            self.lr_scheduler.step()  

            if epoch < 1:
                continue

            # Run testing more frequently
            test_frequency = 3  # Test every 3 epochs
            save_frequency = 2  # Save checkpoints every 2 epochs

            # Always save checkpoint after each epoch
            if epoch % save_frequency == 0:
                # Save checkpoint with model, optimizer and lr_scheduler states
                self.save_checkpoint(epoch, self.min_eer, self.model_state)

            if epoch % test_frequency == 0:
                # --------------- test --------------- #
                eer = self.trainer.test(epoch)

                # --------------- eer check and save --------------- #
                if self.min_eer is None:
                    self.min_eer = eer
                    for test_type in self.min_eer.keys():
                        self.model_state[test_type] = self.model.state_dict()
                else:
                    for test_type in self.min_eer.keys():
                        # Save best model if improved
                        if eer[test_type] < self.min_eer[test_type]:
                            self.min_eer[test_type] = eer[test_type]
                            self.model_state[test_type] = self.model.state_dict()

                            # Save best model checkpoint with full state
                            self.save_checkpoint(
                                epoch, 
                                self.min_eer, 
                                self.model_state, 
                                is_best=True, 
                                test_type=test_type
                            )

                # --------------- log and schedule learning rate --------------- #
                print(f"epoch: {epoch} \neer: {eer} \nmin_eer:{self.min_eer}")        


if __name__ == '__main__':
    import time
    program = Main()
    program.start()
    wandb.finish()
    print("Training finished.")
