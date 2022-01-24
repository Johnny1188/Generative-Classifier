import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import numpy as np
import math

def show_img(imgs, titles=None):
    fig, axes = plt.subplots(
        ((imgs.shape[0]-1) // 5) + 1, 5 if (len(imgs.shape) == 4 and imgs.shape[0] > 1) else 1,
        squeeze=False,
        figsize=(20, 4 * ((imgs.shape[0]-1) // 5 + 1))
    )
    curr_img_i = 0
    curr_img = imgs if len(imgs.shape) == 3 else imgs[curr_img_i]
    while curr_img != None:
        axes[curr_img_i // 5, curr_img_i % 5].imshow(np.transpose(curr_img, (1, 2, 0)))
        if type(titles) == str: axes[curr_img_i // 5, curr_img_i % 5].set_title(titles)
        if type(titles) in (list, tuple, set, torch.tensor): axes[curr_img_i // 5, curr_img_i % 5].set_title(titles[curr_img_i])
        curr_img_i += 1
        curr_img = None if (len(imgs.shape) == 3 or curr_img_i >= imgs.shape[0]) else imgs[curr_img_i]
    plt.show()

def plot_loss_history(loss_history):
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(25,8))
    ax[0].plot(loss_history["classifier"])
    ax[0].title.set_text("classifier loss")
    ax[1].plot(loss_history["generator"])
    ax[1].title.set_text("generator loss")

def show_samples(samples, epoch=0, from_sample_num=0, num_of_samples_to_show=5, classes=None):
    titles_gt = titles_preds = None
    if classes:
        titles_gt = [f"Ground truth: {classes[samples[epoch][2][from_sample_num+i]]}" for i in range(num_of_samples_to_show)]
        titles_preds = [f"Reconstructed + pred.: {classes[torch.argmax(samples[epoch][3][from_sample_num+i])]}" for i in range(num_of_samples_to_show)]

    show_img(
        samples[epoch][0][from_sample_num:from_sample_num+num_of_samples_to_show], 
        titles_gt
    )
    show_img(
        samples[epoch][1][from_sample_num:from_sample_num+num_of_samples_to_show], 
        titles_preds
    )

def plot_weights(weights):
    fig, ax = plt.subplots(figsize=(30,9))
    sns.heatmap(
        weights,
        xticklabels=15,
        axes=ax
    )
    ax.set_ylabel("layer n-1 neurons")
    ax.set_xlabel("layer n neurons")

def plot_gradual_classification_loss(classification_loss_history, n_cols=3):
    n_batches = len(classification_loss_history)

    _ = plt.figure(1, figsize=(14,8))
    for b_i in range(n_batches):
        plt.plot(classification_loss_history[b_i])
    plt.show()

    _ = plt.figure(2)
    n_rows = math.ceil(n_batches / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(min(23, n_batches*6),(n_batches+n_cols//n_cols)*2 - 3))
    for i in range(n_batches):
        if n_rows > 1:
            axes[i // n_cols][i % n_cols].plot(classification_loss_history[i])
        else:
            axes[i % n_cols].plot(classification_loss_history[i])
    plt.show()

def plot_conv_channels(conv_channels, n_rows=8):
    # conv_channels : torch.Tensor N x W x H (N = num of conv. channels, W = width, H = height)
    conv_channels = conv_channels[:,None,:,:] # expand dim
    conv_channels_grid = torchvision.utils.make_grid(conv_channels, nrow=n_rows)
    plt.figure(figsize=(20,(conv_channels.shape[0]//n_rows)*3))
    plt.imshow(conv_channels_grid.permute(1, 2, 0))

    return(None)
