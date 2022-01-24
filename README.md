# Generative classifier *(experiment)*
**Main inspiration:** Recurrent communication in the brain between lower-level visual areas and prefrontal cortex (among others) during task-related visual processing + [Adaptive resonance theory (Stephen Grossberg and Gail Carpenter)](http://www.scholarpedia.org/article/Adaptive_resonance_theory).

**Hypothesis:** Classification at inference time based primarily on recurrent generative process should yield superior results. Incorrectly categorized objects should be harder to reconstruct.

**Quick sketch of the architecture:**

![Sketch of the architecture](/results/arch_sketch.jpg)


## Results
**Improvement thanks to the generative process on Fashion MNIST dataset with 50 % of the pixels zeroed-out:** 
*(from baseline being the neural network on the left side on the sketch above - comparison not 100 % fair, generative process requires more parameters)*
![Improvement](/results/improvement_from_generative_process.png)

**Learned categories of MNIST by the generative process after 1 epoch:** *(with zeroed-out input from the classfier's CNN block, only one-hot encoded category on the input)*
![Learned categories](/results/mnist_after_1_epoch.jpg)
