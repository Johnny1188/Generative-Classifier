import torch
from torch import nn
import wandb

from Classifier import ClassifierHeadBlock, ClassifierCNNBlock
from Generator import GeneratorHeadBlock, GeneratorCNNBlock

class ReGALModel(nn.Module):
  def __init__(self, config_dict):
    assert config_dict["generator_cnn_block_in_layer_shapes"][0] == config_dict["classifier_cnn_output_dim"]
    assert config_dict["generator_prediction_in_layer_shapes"][0] == config_dict["classifier_head_layers"][-1]
    assert config_dict["generator_in_combined_main_layer_shapes"][0] == config_dict["generator_cnn_block_in_layer_shapes"][-1]+config_dict["generator_prediction_in_layer_shapes"][-1]
    super().__init__()

    self.generator_cnn_input_dims = config_dict["generator_input_dims"]
    self.classifier_cnn_output_dim_flattened = config_dict["classifier_cnn_output_dim"]
    self.device = config_dict["device"]

    self.classifier = {
        "cnn_block": ClassifierCNNBlock(
          cnn_layers=config_dict["classifier_cnn_layers"],
          input_dims=config_dict["classifier_cnn_input_dims"],
          verbose=config_dict["verbose"]
        ),
        "head_block": ClassifierHeadBlock(
          layer_shapes=config_dict["classifier_head_layers"],
          input_dim=config_dict["classifier_cnn_output_dim"],
          verbose=config_dict["verbose"]
        )
    }

    self.generator = {
        "head_block": GeneratorHeadBlock(
          classifier_cnn_block_in_layer_shapes=config_dict["generator_cnn_block_in_layer_shapes"],
          classifier_prediction_in_layer_shapes=config_dict["generator_prediction_in_layer_shapes"],
          main_layer_shapes=config_dict["generator_in_combined_main_layer_shapes"],
          verbose=config_dict["verbose"]
        ),
        "trans_cnn_block": GeneratorCNNBlock(
          cnn_transpose_layers=config_dict["generator_cnn_trans_layer_shapes"],
          input_dims=config_dict["generator_input_dims"],
          verbose=config_dict["verbose"]
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
    self.to(self.device)
    self.classifier['cnn_block'].to(self.device)
    self.classifier['head_block'].to(self.device)
    self.generator['head_block'].dense_layers_stack_dict["in_classifier_cnn_block"].to(self.device)
    self.generator['head_block'].dense_layers_stack_dict["in_classifier_prediction"].to(self.device)
    self.generator['head_block'].dense_layers_stack_dict["in_combined_main_stack"].to(self.device)
    self.generator['trans_cnn_block'].to(self.device)

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

  def forward(self, X, max_reconstruction_steps=10, batch_size=32):
    classifier_cnn_block_optimizer = torch.optim.Adam(
      self.classifier["cnn_block"].parameters(),
      lr=self._eval_run_classifier_cnn_block_optimizer_lr,
      weight_decay=self._eval_run_classifier_cnn_block_optimizer_weight_decay
    )
    classifier_head_block_optimizer = torch.optim.Adam(
      self.classifier["head_block"].parameters(),
      lr=self._eval_run_classifier_head_block_optimizer_lr,
      weight_decay=self._eval_run_classifier_head_block_optimizer_weight_decay
    )

    if max_reconstruction_steps == 0: # No reconstruction regularizer
      z = self.classifier["cnn_block"](X)
      z = z.reshape((batch_size, self.classifier_cnn_output_dim_flattened))
      y_hat = self.classifier["head_block"](z)
      return(y_hat)

    for reconstruction_step in range(max_reconstruction_steps):
      z = self.classifier["cnn_block"](X)
      z = z.reshape((batch_size, self.classifier_cnn_output_dim_flattened))
      y_hat = self.classifier["head_block"](z)

      h = self.generator["head_block"](z, y_hat)
      h_reshaped_for_cnn_block = torch.reshape(h, (batch_size, *self.generator_cnn_input_dims))
      X_hat = self.generator["trans_cnn_block"](h_reshaped_for_cnn_block)
      # X_hat = nn.functional.pad(X_hat, (3, 3, 3, 3))
      # X_hat = nn.functional.pad(X_hat, (0, 1, 0, 1))

      reconstruction_loss = self.loss_func_img_reconstruction(X_hat, X)
      reconstruction_loss.backward()

      # for l in self.classifier["head_block"].dense_layers_stack:
      #   if type(l) == torch.nn.modules.linear.Linear:
      #     print(torch.sum( torch.abs(l.weight.grad) ))

      classifier_cnn_block_optimizer.step()
      classifier_cnn_block_optimizer.zero_grad()

      classifier_head_block_optimizer.step()
      classifier_head_block_optimizer.zero_grad()

      del z, h, h_reshaped_for_cnn_block, X_hat
      if reconstruction_step != max_reconstruction_steps-1:
        del y_hat
      # print(f"#{reconstruction_step}:\n{torch.argmax(y_hat, dim=1)}")
    
    return(y_hat)
  
  def turn_components_to_eval_mode(self):
    self.classifier['cnn_block'].eval()
    self.classifier['head_block'].eval()

    self.generator['head_block'].dense_layers_stack_dict["in_classifier_cnn_block"].eval()
    self.generator['head_block'].dense_layers_stack_dict["in_classifier_prediction"].eval()
    self.generator['head_block'].dense_layers_stack_dict["in_combined_main_stack"].eval()
    self.generator['trans_cnn_block'].eval()

  def load_pretrained_params(self, filepath, load_optimizers=False):
    checkpoint = torch.load(filepath, map_location=self.device)
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

  def pretrain(self, epochs, X_train_loader, batch_size=32, past_loss_history=None, verbose=True, wandb_run=False):
    loss_history = {"classifier":[],"generator":[]} if past_loss_history == None else past_loss_history
    samples =[]

    for epoch_i in range(epochs):
      for data in X_train_loader:
        X, y = [part_of_data.to(self.device) for part_of_data in data]

        if( X.shape[0] != batch_size ):
            print("Incorrect batch size!")
            continue
        
        # -----
        # Classification step (supervised)
        z = self.classifier["cnn_block"](X)
        z = z.reshape((batch_size, self.classifier_cnn_output_dim_flattened))
        y_hat = self.classifier["head_block"](z)
        classification_loss = self.loss_func_classification(y_hat, y)
        classification_loss.backward(retain_graph=True)
        self.classifier_optimizer.step()
        self.classifier_optimizer.zero_grad()
        # -----


        # -----
        # Reconstruction step w/ the true target labels
        # y_onehot = nn.functional.one_hot(y,10)
        # h = self.generator["head_block"](z.detach(), y_onehot.detach())
        # -----


        # -----
        # Reconstruction step w/ the predicted target labels (can be used in an unsupervised mode) START
        h = self.generator["head_block"](z.detach(), y_hat.detach())
        h_reshaped_for_cnn_block = torch.reshape(h, (batch_size, *self.generator_cnn_input_dims))
        X_hat = self.generator["trans_cnn_block"](h_reshaped_for_cnn_block)
        reconstruction_loss = self.loss_func_img_reconstruction(X_hat, X)
        reconstruction_loss.backward()
        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad()
        # Pretraining w/ the predicted target labels END
        # -----


        # -----
        # Dynamically change the learning rate according to the confidence of classifier network
        # for g in optimizer.param_groups:
        #   g['lr'] = 0.001
        # -----


        # -----
        # Another gradient step for the generator 
        # - classification accuracy of the classifier as it is looking at the reconstructed image
        #   z = self.classifier["cnn_block"](X_hat.detach())
        #   z = z.reshape((BATCH_SIZE, self.classifier_cnn_output_dim_flattened))
        #   y_hat = self.classifier["head_block"](z)
        #   classification_loss_from_reconstruction = nn.functional.cross_entropy(y_hat, y)
        #   classification_loss_from_reconstruction.backward(retain_graph=True)
        #   reconstruction_optimizer.step()
        #   reconstruction_optimizer.zero_grad()
        # -----

      loss_history["classifier"].append(classification_loss.detach().cpu())
      loss_history["generator"].append(reconstruction_loss.detach().cpu())
      samples.append((X.detach().cpu(), X_hat.detach().cpu(), y.detach().cpu(), y_hat.detach().cpu()))
      if verbose:
        print(f"Epoch: [{epoch_i+1}/{epochs}] >>> Classification loss: {round(classification_loss.detach().cpu().item(),4)} >>> Reconstruction loss: {round(reconstruction_loss.detach().cpu().item(),4)}")
        if wandb_run:
          wandb.log({
            "classification_loss": round(classification_loss.detach().cpu().item(),4),
            "reconstruction_loss": round(reconstruction_loss.detach().cpu().item(),4) 
          })

    return(loss_history, samples)
