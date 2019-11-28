# W1 Estimation
W1(P,Q) estimation given samples from two distributions P, Q. For example, to estimate W1 distance between the ImageNet train classes of 'tabby cat' and 'tiger' (each of which have 1300 images), we can run the following:

```shell
python estimate_w1.py --dataroot1 data/tabby_cat_n02123045/ --dataroot2 data/tiger_n02129604/ \
--batchSize 50 --nepochs 25 --cuda
```

It is assumed that the `dataroot` directories will be flat and contain only images. The output plot of W1 estimate versus epoch number is shown below:

![Convergence](https://i.imgur.com/Rd0yDGn.png)

The approximate W1 distance in this case was 1.46, and from the above plot it seems as though the estimator has roughly converged after 100 epochs.

Note this code uses the MLP and DCGAN critic from [Martin Arjovsky's repository here](https://github.com/martinarjovsky/WassersteinGAN/), and also note that this code uses clipping to maintain a k-Lipschitz constraint (the discriminator/critic is not strictly 1-Lipschitz like it should be in the Kantorovich formulation).
