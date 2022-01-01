import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def imshow_mnist(img, title):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).reshape(-1,npimg.shape[-1]))
    plt.title(title)
    plt.show()

def imshow_cifar10(imgs, titles=None, w_color=True):
    fig, axes = plt.subplots(
        ((imgs.shape[0]-1) // 5) + 1, 5 if (len(imgs.shape) == 4 and imgs.shape[0] > 1) else 1,
        squeeze=False,
        figsize=(20, 4 * ((imgs.shape[0]-1) // 5 + 1))
    )
    curr_img_i = 0
    curr_img = imgs if len(imgs.shape) == 3 else imgs[curr_img_i]
    while curr_img != None:
        if w_color:
            axes[curr_img_i // 5, curr_img_i % 5].imshow(np.transpose(curr_img, (1, 2, 0)))
        else:
            axes[curr_img_i // 5, curr_img_i % 5].imshow(np.transpose(curr_img, (1, 2, 0)).reshape(-1,curr_img.shape[-1]))
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

    imshow_cifar10(
        samples[epoch][0][from_sample_num:from_sample_num+num_of_samples_to_show], 
        titles_gt,
        w_color=True
    )
    imshow_cifar10(
        samples[epoch][1][from_sample_num:from_sample_num+num_of_samples_to_show], 
        titles_preds,
        w_color=True
    )

def plot_weights(weights):
    fig, ax = plt.subplots(figsize=(30,6))
    sns.heatmap(
        weights,
        xticklabels=15,
        axes=ax
    )
    ax.set_ylabel("layer n-1 neurons")
    ax.set_xlabel("layer n neurons")
