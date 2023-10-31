# Generative image classifier *(experiment)*

[Inspiration](#inspiration) - [Hypothesis](#hypothesis) - [Results](#results)

---


## Inspiration
Recurrent communication in the brain between lower-level visual areas and prefrontal cortex (among others) during task-related visual processing + [Adaptive resonance theory (Stephen Grossberg and Gail Carpenter)](http://www.scholarpedia.org/article/Adaptive_resonance_theory).

## Hypothesis
Classification at inference time based primarily on the recurrent generative process should yield superior results. Incorrectly categorized objects should be harder to reconstruct.

**Sketch of the architecture:**

![Sketch of the architecture](/results/architecture_sketch.jpg)


## Results
**Improvement thanks to the generative process on the Fashion MNIST dataset with 50% of the pixels zeroed out** (baseline being the neural network on the left side on the sketch above; comparison not 100% fair, the generative process requires additional parameters):

![Improvement](/results/improvement_from_generative_process.png)

**Learned categories of MNIST by the generative process after 1 epoch** (the generator with zeroed-out input from the classifier's CNN block, only one-hot encoded category on the input):

![Learned categories](/results/mnist_after_1_epoch.jpg)

More complex datasets such as CIFAR10 showed only small improvements with additional computational overhead resulting in slower training and inference. In the future, I would like to experiment with the generative model reconstructing the activations of the classifier or some other latent variables, thereby reducing the dependence on the dataset.
