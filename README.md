## The Generalized Affine Model (GAM)

The GAM is a latent variable model developed to study large populations of simultaneously recorded neurons. It incorporates latent variables that can either add to or multiply the stimulus response of individual neurons. This model is a generalization of several other models that have appeared in the computational neuroscience literature, including:
- [Lin *et al*, The nature of shared cortical variability (Neuron 2015)](http://www.sciencedirect.com/science/article/pii/S089662731500598X)
- [Rabinowitz *et al*, Attention stabilizes the shared gain of V4 populations (Elife 2015)](https://elifesciences.org/articles/08998#abstract)
- [Arandia-Romero *et al*, Multiplicative and additive modulation of neuronal tuning with population activity affects encoded information (Neuron 2016)](http://www.sciencedirect.com/science/article/pii/S089662731600091X)

A more thorough explanation of the modeling framework can be found in our preprint on biorxiv: [Characterizing the nonlinear structure of shared variability in cortical neuron populations using latent variable models](https://www.biorxiv.org/content/early/2018/09/30/407858)


The `doc` directory contains scripts that show how to use the model on several simulated datasets (coming soon).

The GAM optimizes model parameters using [Mark Schmidt's](http://www.cs.ubc.ca/~schmidtm/) minFunc package, which is located in the `lib` directory and should work out of the box. If not you may need to run the mexAll.m file from the `lib/minFunc_2012` directory.
