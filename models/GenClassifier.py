import torch
from torch import nn

from models.Classifier import ClassifierHeadBlock, ClassifierCNNBlock
from models.Generator import GeneratorHeadBlock, GeneratorCNNBlock


class GenClassifier(nn.Module):
    def __init__(self, config_dict):
        assert config_dict["generator_cnn_block_in_layer_shapes"][0] == config_dict["classifier_cnn_output_dim"]
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
                input_dims=config_dict["generator_trans_cnn_input_dims"]
            )
        }

        self._init_optimizers(config_dict["classifier_lr"], config_dict["classifier_weight_decay"], config_dict["generator_lr"], config_dict["generator_weight_decay"])
        self._move_to_target_device()
        self._eval_run_classifier_cnn_block_optimizer_lr = config_dict["eval_run_classifier_cnn_block_optimizer_lr"]
        self._eval_run_classifier_cnn_block_optimizer_weight_decay = config_dict["eval_run_classifier_cnn_block_optimizer_weight_decay"]
        self._eval_run_classifier_head_block_optimizer_lr = config_dict["eval_run_classifier_head_block_optimizer_lr"]
        self._eval_run_classifier_head_block_optimizer_weight_decay = config_dict["eval_run_classifier_head_block_optimizer_weight_decay"]
    
    def forward(self, X, hypotheses_testing=True, num_of_reconstruction_steps=2, batch_size=32, norm_cos_similarities=False, cos_similarity_multiplier=1.0, _y=None):
        # (tracking the continual change in classification loss when reconstruction regularizer is used)
        z_real = torch.flatten(self.classifier["cnn_block"](X), start_dim=1)
        y_hat_initial = self.classifier["head_block"](z_real.detach())
        if(hypotheses_testing == False): return(y_hat_initial) # the baseline prediction by the classifier from the input image

        cos_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_similarities = torch.zeros_like(y_hat_initial)

        for hypothesis_i in range(self.config_dict["classifier_head_layers"][-1]):
            hypothesized_category_onehot = torch.zeros((batch_size, self.config_dict["classifier_head_layers"][-1]), device=self.config_dict["device"])
            hypothesized_category_onehot[:,hypothesis_i] = 1.

            h = self.generator["head_block"](z_real.detach(), hypothesized_category_onehot)
            h_reshaped_for_cnn_block = torch.reshape(h, (batch_size, *self.config_dict["generator_trans_cnn_input_dims"]))
            X_hat = self.generator["trans_cnn_block"](h_reshaped_for_cnn_block)

            for _ in range(num_of_reconstruction_steps): # let the generator transform the z based on the hypothesized category
                z = self.classifier["cnn_block"](X_hat).reshape((batch_size, self.config_dict["classifier_cnn_output_dim"]))
                h = self.generator["head_block"](z.detach(), hypothesized_category_onehot)
                h_reshaped_for_cnn_block = torch.reshape(h, (batch_size, *self.config_dict["generator_trans_cnn_input_dims"]))
                X_hat = self.generator["trans_cnn_block"](h_reshaped_for_cnn_block)

            z_gen = self.classifier["cnn_block"](X_hat).reshape((batch_size, self.config_dict["classifier_cnn_output_dim"]))
            y_hat_from_gen = self.classifier["head_block"](z_gen)

            cos_similarity = cos_func(z_real, z_gen)
            if norm_cos_similarities: cos_similarity /= cos_similarity.norm()
            cos_similarity = torch.exp(cos_similarity * cos_similarity_multiplier) * nn.functional.softmax(y_hat_from_gen,dim=1)[:,hypothesis_i]
            cos_similarities[:, hypothesis_i] = cos_similarity

        # print(f"{_y[0].item()}\nBefore: {[round(n,4) for n in nn.functional.softmax(y_hat_initial[0],dim=0).tolist()]}\nAfter:  {[round(n,4) for n in nn.functional.softmax(cos_similarities[0],dim=0).tolist()]}\n")
        y_hat_final = y_hat_initial + (nn.functional.softmax(cos_similarities,dim=1).T * y_hat_initial.std(dim=1)).T
        return(y_hat_final)

    def _move_to_target_device(self):
        self.to(self.config_dict["device"])
        self.classifier['cnn_block'].to(self.config_dict["device"])
        self.classifier['head_block'].to(self.config_dict["device"])
        self.generator['head_block'].dense_layers_stack_dict["in_classifier_cnn_block"].to(self.config_dict["device"])
        self.generator['head_block'].dense_layers_stack_dict["in_classifier_prediction"].to(self.config_dict["device"])
        self.generator['head_block'].dense_layers_stack_dict["in_combined_main_stack"].to(self.config_dict["device"])
        self.generator['trans_cnn_block'].to(self.config_dict["device"])

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
