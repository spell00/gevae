# Variational Autoencoder for Gene Expression

Deep learning models are very powerful models, but their very high capacity makes them prone to overfitting. Data 
augmentation is a common strategy used to reduce overfitting of Deep Learning models. For example, it is frequently 
used in computer vision. However, data augmentation works better if it can make use of the data structure; in computer 
vision, it often involves a combination of rotations and distortions of the images to name just a few possibilities. 
Gene expression data can't be as easily augmented, because the structure is not as obvious. 

Deep Generative Models (DGMs), such as Generative Adverserial Networks (GANs) or Variational Auto-Encoders (VAEs) might 
be useful for data augmentation of Gene Expression datasets. If a DGM can learn a good enough representation of the 
data, it could be used to generate new samples or slightly modify the samples in a way that incorporate Gaussian noise 
each time a sample is used. The objective is to obtain a new highly probable sample that was not in the original dataset.

Alternatively, VAEs can be used for dimentionality reduction.

The data can be obtained here : https://www.kaggle.com/c/lish-moa/data

The training data consist of the expression from ~750 genes and ~100 cell viability data from 3982 samples. 

Sources:

1- Auto-Encoding Variational Bayes (Variational Auto-Encoders) : https://arxiv.org/abs/1312.6114