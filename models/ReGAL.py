import torch
from torch import nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

from models.Classifier import ClassifierHeadBlock, ClassifierCNNBlock
from models.Generator import GeneratorHeadBlock, GeneratorCNNBlock
from utils.train_logging import PretrainLogging

class ReGALModel(nn.Module):
    def __init__(self, config_dict):
        assert config_dict["generator_cnn_block_in_layer_shapes"][0] == config_dict["classifier_cnn_output_dim"]
        # assert config_dict["generator_prediction_in_layer_shapes"][0] == config_dict["classifier_head_layers"][-1]
        assert config_dict["generator_in_combined_main_layer_shapes"][0] == config_dict["generator_cnn_block_in_layer_shapes"][-1]+config_dict["generator_prediction_in_layer_shapes"][-1]
        super().__init__()

        self.config_dict = config_dict

        self.classifier = {
            "cnn_block": ClassifierCNNBlock(
                cnn_layers=config_dict["classifier_cnn_layers"],
                input_dims=config_dict["classifier_cnn_input_dims"]
            ),
            "head_block": ClassifierHeadBlock(
                layer_shapes=config_dict["classifier_head_layers"],
                input_dim=config_dict["classifier_cnn_output_dim"]
            )
        }

        self.generator = {
            "head_block": GeneratorHeadBlock(
                classifier_cnn_block_in_layer_shapes=config_dict["generator_cnn_block_in_layer_shapes"],
                classifier_prediction_in_layer_shapes=config_dict["generator_prediction_in_layer_shapes"],
                main_layer_shapes=config_dict["generator_in_combined_main_layer_shapes"],
                num_of_classes=config_dict["classifier_head_layers"][-1]
            ),
            "trans_cnn_block": GeneratorCNNBlock(
                cnn_transpose_layers=config_dict["generator_cnn_trans_layer_shapes"],
                input_dims=config_dict["generator_input_dims"]
            )
        }

        self._init_loss_funcs()
        self._init_optimizers(config_dict["classifier_lr"], config_dict["classifier_weight_decay"], config_dict["generator_lr"], config_dict["generator_weight_decay"])
        self._move_to_target_device()
        self._eval_run_classifier_cnn_block_optimizer_lr = config_dict["eval_run_classifier_cnn_block_optimizer_lr"]
        self._eval_run_classifier_cnn_block_optimizer_weight_decay = config_dict["eval_run_classifier_cnn_block_optimizer_weight_decay"]
        self._eval_run_classifier_head_block_optimizer_lr = config_dict["eval_run_classifier_head_block_optimizer_lr"]
        self._eval_run_classifier_head_block_optimizer_weight_decay = config_dict["eval_run_classifier_head_block_optimizer_weight_decay"]

    def _move_to_target_device(self):
        self.to(self.config_dict["device"])
        self.classifier['cnn_block'].to(self.config_dict["device"])
        self.classifier['head_block'].to(self.config_dict["device"])
        self.generator['head_block'].dense_layers_stack_dict["in_classifier_cnn_block"].to(self.config_dict["device"])
        self.generator['head_block'].dense_layers_stack_dict["in_classifier_prediction"].to(self.config_dict["device"])
        self.generator['head_block'].dense_layers_stack_dict["in_combined_main_stack"].to(self.config_dict["device"])
        self.generator['trans_cnn_block'].to(self.config_dict["device"])

    def _init_loss_funcs(self):
        self.loss_func_classification = nn.CrossEntropyLoss()
        self.loss_func_img_reconstruction = nn.MSELoss()

    def _init_optimizers(self, classifier_lr, classifier_weight_decay, generator_lr, generator_weight_decay):
        self.classifier_optimizer = torch.optim.Adam(
            [*self.classifier["head_block"].parameters(),
            *self.classifier["cnn_block"].parameters()],
            lr=classifier_lr,
            weight_decay=classifier_weight_decay
        )
        self.generator_optimizer = torch.optim.Adam(
            [*self.generator["head_block"].dense_layers_stack_dict["in_classifier_cnn_block"].parameters(),
            *self.generator["head_block"].dense_layers_stack_dict["in_classifier_prediction"].parameters(),
            *self.generator["head_block"].dense_layers_stack_dict["in_combined_main_stack"].parameters(),
            *self.generator["trans_cnn_block"].parameters()],
            lr=generator_lr,
            weight_decay=generator_weight_decay
        )

    def forward(self, X, max_reconstruction_steps=10, batch_size=32, y=None):
        # parameter y (targets) only for logging purposes
        # (tracking the continual change in classification loss when reconstruction regularizer is used)

        if max_reconstruction_steps == 0: # No reconstruction regularizer
            z = self.classifier["cnn_block"](X)
            z = z.reshape((batch_size, self.config_dict["classifier_cnn_output_dim"]))
            y_hat = self.classifier["head_block"](z)
            return(y_hat)

        ##### with iterative reconstruction #####
        # classifier_cnn_block_optimizer = torch.optim.Adam(
        #     self.classifier["cnn_block"].parameters(),
        #     lr=self._eval_run_classifier_cnn_block_optimizer_lr,
        #     weight_decay=self._eval_run_classifier_cnn_block_optimizer_weight_decay
        # )
        classifier_head_block_optimizer = torch.optim.Adam(
            self.classifier["head_block"].parameters(),
            lr=self._eval_run_classifier_head_block_optimizer_lr,
            weight_decay=self._eval_run_classifier_head_block_optimizer_weight_decay
        )

        # print(f"#####\nTargets >>> {y}")
        pred_loss_during_reconstruction = []
        for reconstruction_step in range(max_reconstruction_steps):
            z = self.classifier["cnn_block"](X)
            z = z.reshape((batch_size, self.config_dict["classifier_cnn_output_dim"]))
            y_hat = self.classifier["head_block"](z)

            h = self.generator["head_block"](z, y_hat)
            h_reshaped_for_cnn_block = torch.reshape(h, (batch_size, *self.config_dict["generator_input_dims"]))
            X_hat = self.generator["trans_cnn_block"](h_reshaped_for_cnn_block)
            # X_hat = nn.functional.pad(X_hat, (3, 3, 3, 3))
            # X_hat = nn.functional.pad(X_hat, (0, 1, 0, 1))

            reconstruction_loss = self.loss_func_img_reconstruction(X_hat, X)
            reconstruction_loss.backward()

            # classifier_cnn_block_optimizer.step()
            # classifier_cnn_block_optimizer.zero_grad()

            classifier_head_block_optimizer.step()
            # for l in self.classifier["head_block"].dense_layers_stack:
            #     if type(l) == torch.nn.modules.linear.Linear:
            #         print(torch.sum(torch.abs(l.weight.grad)))
            classifier_head_block_optimizer.zero_grad()

            if y != None:
                pred_loss_during_reconstruction.append(self.loss_func_classification(y_hat,y).detach().cpu().item())

            del z, h, h_reshaped_for_cnn_block, X_hat
            # print(f"[{reconstruction_step+1:02}/{max_reconstruction_steps}] >>> {torch.argmax(y_hat, dim=1).detach().cpu().tolist()}")
            if reconstruction_step != max_reconstruction_steps-1:
                del y_hat

        return(y_hat, pred_loss_during_reconstruction)

    def turn_model_to_mode(self, mode="train"):
        assert mode in ("train", "training", "eval", "evaluation")
        train_bool = True if mode in ("train", "training") else False

        self.classifier['cnn_block'].train(train_bool)
        self.classifier['head_block'].train(train_bool)

        self.generator['head_block'].dense_layers_stack_dict["in_classifier_cnn_block"].train(train_bool)
        self.generator['head_block'].dense_layers_stack_dict["in_classifier_prediction"].train(train_bool)
        self.generator['head_block'].dense_layers_stack_dict["in_combined_main_stack"].train(train_bool)
        self.generator['trans_cnn_block'].train(train_bool)
        self.train(train_bool)

    def load_pretrained_params(self, filepath, load_optimizers=False):
        checkpoint = torch.load(filepath, map_location=self.config_dict["device"])
        self.classifier['cnn_block'].load_state_dict(checkpoint['classifier_cnn_block'])
        self.classifier['head_block'].load_state_dict(checkpoint['classifier_head_block'])

        self.generator['head_block'].dense_layers_stack_dict["in_classifier_cnn_block"].load_state_dict(checkpoint['generator_head_block_in_classifier_cnn_block'])
        self.generator['head_block'].dense_layers_stack_dict["in_classifier_prediction"].load_state_dict(checkpoint['generator_head_block_in_classifier_prediction'])
        self.generator['head_block'].dense_layers_stack_dict["in_combined_main_stack"].load_state_dict(checkpoint['generator_head_block_in_combined_main_stack'])
        self.generator['trans_cnn_block'].load_state_dict(checkpoint['generator_trans_cnn_block'])

        if load_optimizers:
            self.classifier_optimizer.load_state_dict(checkpoint['classifier_optimizer'])
            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
        return(True)

    def save_model_params(self, filepath):
        a = input(f"Confirm by typing the letter 's' or the word 'save'\n({filepath}):\n")
        if a == "s" or a == "save":
            torch.save({
                'classifier_cnn_block': self.classifier['cnn_block'].state_dict(),
                'classifier_head_block': self.classifier['head_block'].state_dict(),
                'generator_head_block_in_classifier_cnn_block': self.generator['head_block'].dense_layers_stack_dict["in_classifier_cnn_block"].state_dict(),
                'generator_head_block_in_classifier_prediction': self.generator['head_block'].dense_layers_stack_dict["in_classifier_prediction"].state_dict(),
                'generator_head_block_in_combined_main_stack': self.generator['head_block'].dense_layers_stack_dict["in_combined_main_stack"].state_dict(),
                'generator_trans_cnn_block': self.generator['trans_cnn_block'].state_dict(),
                'classifier_optimizer': self.classifier_optimizer.state_dict(),
                'generator_optimizer': self.generator_optimizer.state_dict(),
            }, filepath)
        print(f"Successfully saved the model's parameters to {filepath}")
        return(True)

    def pretrain(self, epochs, X_train_loader, batch_size=32, past_loss_history=None, verbose=True, is_wandb_run=False, class_names=None):
        pretrain_logging = PretrainLogging(
            past_loss_history=past_loss_history, verbose=verbose, is_wandb_run=is_wandb_run,
            class_names=class_names, num_of_epochs=epochs
        )

        self.turn_model_to_mode(mode="train")

        for epoch_i in range(epochs):
            for data in X_train_loader:
                X, y = [part_of_data.to(self.config_dict["device"]) for part_of_data in data]

                ##### Classifier #####
                # regular supervised classification step
                z = self.classifier["cnn_block"](X)
                z = z.reshape((batch_size, self.config_dict["classifier_cnn_output_dim"]))
                y_hat = self.classifier["head_block"](z)
                classification_loss = self.loss_func_classification(y_hat, y)
                classification_loss.backward(retain_graph=True)
                self.classifier_optimizer.step()
                self.classifier_optimizer.zero_grad()
                #####


                ##### Generator #####
                # 1. Reconstruction step w/ the classes (either w/ predicted => unsupervised mode; or w/ true target labels => only supervised)
                y_onehot = nn.functional.one_hot(y,10).float().to(self.config_dict["device"])
                h = self.generator["head_block"](z.detach(), y_onehot) # <-- pretraining generator on the true target labels (less noise)
                # h = self.generator["head_block"](z.detach(), y_hat.detach()) # <-- pretraining generator on classifier's prediction
                h_reshaped_for_cnn_block = torch.reshape(h, (batch_size, *self.config_dict["generator_input_dims"]))
                X_hat = self.generator["trans_cnn_block"](h_reshaped_for_cnn_block)
                reconstruction_loss = self.loss_func_img_reconstruction(X_hat, X)

                # 2. Generator's loss for the classification inaccuracy from its reconstructed imgs by the classifier
                z = self.classifier["cnn_block"](X_hat)
                z = z.reshape((self.config_dict["classifier_cnn_input_dims"][0], self.config_dict["classifier_cnn_output_dim"]))
                y_hat = self.classifier["head_block"](z)
                classification_loss_from_reconstructed_img = self.loss_func_classification(y_hat, y)

                merged_generator_loss = \
                    self.config_dict["generator_alpha"] * reconstruction_loss \
                    + (1 - self.config_dict["generator_alpha"]) * classification_loss_from_reconstructed_img
                merged_generator_loss.backward()
                self.classifier_optimizer.zero_grad()
                
                self.generator_optimizer.step()
                self.generator_optimizer.zero_grad()
                #####

            pretrain_logging.track_epoch(
                epoch_i, classification_loss.detach().cpu().item(), reconstruction_loss.detach().cpu().item(),
                X.detach().cpu(), X_hat.detach().cpu(), y.detach().cpu(), y_hat.detach().cpu()
            )

        loss_history, samples = pretrain_logging.end_pretraining()
        return(loss_history, samples)
