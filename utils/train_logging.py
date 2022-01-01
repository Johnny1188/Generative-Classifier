import os
import shutil
import matplotlib
# import numpy as np
import wandb
import torch

class PretrainLogging():
    def __init__(self, past_loss_history=None, verbose=True, is_wandb_run=False, class_names=None, num_of_epochs=0, path_to_tmp_folder="_regal_tmp"):
        assert (is_wandb_run and class_names) or not is_wandb_run
        self.loss_history = {"classifier":[],"generator":[]} if past_loss_history == None else past_loss_history
        self.samples =[]
        self.verbose = verbose
        self.is_wandb_run = is_wandb_run
        self.class_names = class_names
        self.num_of_epochs = num_of_epochs
        self.path_to_tmp_folder = path_to_tmp_folder

        if self.is_wandb_run:
            os.makedirs(self.path_to_tmp_folder, exist_ok=True) # only for W&B logging

    def track_epoch(self, epoch_i, classification_loss, reconstruction_loss, X, X_hat, y, y_hat):
        self.loss_history["classifier"].append(classification_loss)
        self.loss_history["generator"].append(reconstruction_loss)
        self.samples.append((X, X_hat, y, y_hat))
        if self.verbose:
            print(
                f"Epoch: [{epoch_i+1}/{self.num_of_epochs}]",
                f">>> Classification loss: {round(classification_loss,4)}",
                f">>> Reconstruction loss: {round(reconstruction_loss,4)}"
            )
        if self.is_wandb_run:
            # W&B tends to crash when loading and saving imgs from arrays directly
            matplotlib.image.imsave(os.path.join(self.path_to_tmp_folder, "in.png"), X[0].numpy().transpose((1,2,0)))
            matplotlib.image.imsave(os.path.join(self.path_to_tmp_folder, "reconstructed.png"), X_hat[0].numpy().transpose((1,2,0)))
            wandb.log({
                "classification_loss": round(classification_loss, 4),
                "reconstruction_loss": round(reconstruction_loss, 4),
                "img_in": wandb.Image(os.path.join(self.path_to_tmp_folder, "in.png"), caption=f"In ({self.class_names[y[0]]})"),
                "img_reconstructed": wandb.Image(os.path.join(self.path_to_tmp_folder, "reconstructed.png"), caption=f"Reconstructed ({self.class_names[torch.argmax(y_hat[0])]}")
            })

    def end_pretraining(self):
        if self.is_wandb_run:
            shutil.rmtree(self.path_to_tmp_folder, ignore_errors=True)
        return(self.loss_history, self.samples)
