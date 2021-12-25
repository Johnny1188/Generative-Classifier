import matplotlib.pyplot as plt
import torch
import numpy as np

def imshow_mnist(img, title):
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)).reshape(-1,npimg.shape[-1]))
  plt.title(title)
  plt.show()

def imshow_cifar10(imgs, titles=None, w_color=True):
  fig, axes = plt.subplots(((imgs.shape[0]-1) // 5)+1, imgs.shape[0] if len(imgs.shape) == 4 else 1, squeeze=False, figsize=(17,5))
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