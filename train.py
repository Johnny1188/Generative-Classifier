import torch
from torch import nn

from utils.train_logging import PretrainLogging


def train(model, epochs, X_train_loader, batch_size=32, verbose=True, is_wandb_run=False, class_names=None):
    pretrain_logging = PretrainLogging(
        verbose=verbose, is_wandb_run=is_wandb_run,
        class_names=class_names, num_of_epochs=epochs
    )

    model.turn_model_to_mode(mode="train")

    cosine_sim_func = nn.CosineSimilarity(dim=1)
    loss_func_classification = nn.CrossEntropyLoss()
    loss_func_img_reconstruction = nn.MSELoss()

    for epoch_i in range(epochs):
        for data in X_train_loader:
            X, y = [part_of_data.to(model.config_dict["device"]) for part_of_data in data]
            
            ##### Generator #####
            model.generator_optimizer.zero_grad()
            # 1. Reconstruction step w/ the classes (either w/ predicted => unsupervised mode; or w/ true target labels => only supervised)
            y_onehot = nn.functional.one_hot(y, model.config_dict["classifier_head_layers"][-1]).float().to(model.config_dict["device"])
            z = model.classifier["cnn_block"](X).reshape((batch_size, model.config_dict["classifier_cnn_output_dim"]))
            h = model.generator["head_block"](z.detach(), y_onehot) # pretraining generator on the true target labels
            h = torch.reshape(h, (batch_size, *model.config_dict["generator_trans_cnn_input_dims"]))
            X_hat = model.generator["trans_cnn_block"](h)
            reconstruction_loss = loss_func_img_reconstruction(X_hat, X)

            # 2. Generator's loss for the classification inaccuracy from its reconstructed imgs by the classifier
            z_from_reconstructed = model.classifier["cnn_block"](X_hat).reshape((batch_size, model.config_dict["classifier_cnn_output_dim"]))
            y_hat = model.classifier["head_block"](z_from_reconstructed)
            classification_loss_from_reconstructed_img = loss_func_classification(y_hat, y)
            z_similarity_loss = torch.mean(-torch.log(cosine_sim_func(z, z_from_reconstructed)))

            # 3. Generator's contrastive loss between internal representations coming from different categories and same z
            #    (model should push them apart)
            y_permuted = (y + torch.randint_like(y, 1, model.config_dict["classifier_head_layers"][-1]-1)) % model.config_dict["classifier_head_layers"][-1]
            y_onehot_permuted = nn.functional.one_hot(y_permuted, model.config_dict["classifier_head_layers"][-1]).float().to(model.config_dict["device"])
            h_permuted_categories = torch.reshape(
                model.generator["head_block"](z.detach(), y_onehot_permuted), (batch_size, *model.config_dict["generator_trans_cnn_input_dims"]))
            h_true_categories = torch.reshape(
                model.generator["head_block"](z.detach(), y_onehot), (batch_size, *model.config_dict["generator_trans_cnn_input_dims"]))
            contrastive_loss = torch.mean(cosine_sim_func(h_true_categories, h_permuted_categories))

            # 4. Zeroing out the z, feeding the generator only with the category
            h_zeroed_z = model.generator["head_block"](torch.zeros_like(z.detach(),device=model.config_dict["device"]), y_onehot)
            h_zeroed_z = torch.reshape(h, (batch_size, *model.config_dict["generator_trans_cnn_input_dims"]))
            X_hat_w_zeroed_z = model.generator["trans_cnn_block"](h_zeroed_z)
            reconstruction_loss_no_z = loss_func_img_reconstruction(X_hat_w_zeroed_z, X)

            merged_generator_loss = \
                model.config_dict["generator_reconstruction_loss_importance"] * reconstruction_loss \
                + model.config_dict["generator_reconstruction_from_no_z_loss_importance"] * reconstruction_loss_no_z \
                + model.config_dict["generator_classification_loss_importance"] * classification_loss_from_reconstructed_img \
                + model.config_dict["generator_z_similarity_loss_importance"] * z_similarity_loss \
                + model.config_dict["generator_contrastive_loss_importance"] * contrastive_loss
            merged_generator_loss.backward()
            model.generator_optimizer.step()
            #####


            ##### Classifier #####
            model.classifier_optimizer.zero_grad()
            # 1. regular supervised classification step
            z = model.classifier["cnn_block"](X).reshape((batch_size, model.config_dict["classifier_cnn_output_dim"]))
            y_hat = model.classifier["head_block"](z)
            classification_loss = loss_func_classification(y_hat, y)
            classification_loss.backward()
            model.classifier_optimizer.step()
            #####

        pretrain_logging.track_epoch(
            epoch_i, classification_loss.detach().cpu().item(), reconstruction_loss.detach().cpu().item(),
            classification_loss_from_reconstructed_img.detach().cpu().item(),
            X.detach().cpu(), X_hat.detach().cpu(), y.detach().cpu(), y_hat.detach().cpu()
        )

    loss_history, samples = pretrain_logging.end_training()
    return(loss_history, samples)
