from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.autograd as autograd
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from PIL import Image
import numpy as np

import models.dcgan as dcgan
import models.mlp as mlp


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot1', required=True, help='path to dataset folder (raw images) representing samples from real distribution (P)')
    parser.add_argument('--dataroot2', required=True, help='path to dataset folder (raw images) representing samples from fake distribution (Q)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='processing batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters; roughly controls the expressive power of discriminator')
    parser.add_argument('--nepochs', type=int, default=25, help='number of epochs to train for, i.e. number of full passes over the dataset to approximate W1')
    parser.add_argument('--generic_data', action='store_true', help='flag for generic (presumably non-image) data; assumes format is of numpy txt arrays')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--print_every_itr', type=int, default=5, help='print ever number of iterations specified here')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--do_gp', action='store_true', help='do gradient penalty')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not opt.mlp_D and opt.generic_data:
        print('Can only use MLP_D with generic data! Add the --mlp_D flag, or try using image data.')

    # first load file names, will later batch data loading
    onlyfiles_real = [f for f in listdir(opt.dataroot1) if isfile(join(opt.dataroot1, f))]
    onlyfiles_fake = [f for f in listdir(opt.dataroot2) if isfile(join(opt.dataroot2, f))]

    # will load a batch of files, assume the file order is already randomized
    def load_image_batch(f_names, dataroot):
        # pre-processing commands
        mean, stddev = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] # not critical, but conventional
        transform_resize = transforms.Compose([transforms.Resize(opt.imageSize), transforms.CenterCrop(opt.imageSize)])
        transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, stddev)])

        # go through and fill the torch tensor
        im_tens = torch.zeros(opt.batchSize, 3, opt.imageSize, opt.imageSize)

        for i in range(opt.batchSize):
            img_pil = Image.open(dataroot + '/' + f_names[i])
            img_pil_resize = transform_resize(img_pil.copy())
            img = transform_norm(img_pil_resize.copy())
            im_tens[i] = img

        return im_tens

    # will load a batch of generic files, assume they are stored as numpy txt arrays
    def load_generic_batch(f_names, dataroot):
        transform = transforms.ToTensor()

        # go through and fill the torch tensor
        im_tens = []

        for i in range(opt.batchSize):
            data_vec = np.loadtxt(dataroot + '/' + f_names[i]).flatten()
            im_tens.append(transform(data_vec))

        return torch.stack(im_tens)

    # define batch loading function
    load_batch = load_generic_batch if opt.generic_data else load_image_batch

    nz = int(opt.nz)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, 1) # ngpu is 1
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, 1, n_extra_layers) # ngpu is 1
        netD.apply(weights_init)

    print(netD)

    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        netD.cuda()
        one, mone = one.cuda(), mone.cuda()

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)

    # calculate gradient penalty
    def calc_gradient_penalty(netD, real_data, fake_data):
        alpha = torch.rand(opt.batchSize, 1)
        alpha = alpha.expand(opt.batchSize, int(real_data.nelement()/opt.batchSize)).contiguous().view(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        alpha = alpha.cuda() if opt.cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if opt.cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                    disc_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10 # LAMBDA=10 hyperparam
        return gradient_penalty

    epoch_hist = []
    for epoch in range(opt.nepochs):
        print(f'\nrunning epoch {epoch} of W1 estimation...')

        # shuffle files
        random.shuffle(onlyfiles_real)
        random.shuffle(onlyfiles_fake)

        # assume limited by smaller dataset for now
        per_epoch_iters = min(len(onlyfiles_real), len(onlyfiles_fake)) // opt.batchSize
        for i in range(per_epoch_iters):
            if i % opt.print_every_itr == 0:
                print(f'epoch {epoch}: {i}/{per_epoch_iters} iterations')

            # clamp parameters to a cube
            if not opt.do_gp:
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            # clear gradients from previous backward
            netD.zero_grad()

            # get the data
            real_data = load_batch(onlyfiles_real[i*opt.batchSize:(i+1)*opt.batchSize], opt.dataroot1)
            fake_data = load_batch(onlyfiles_fake[i*opt.batchSize:(i+1)*opt.batchSize], opt.dataroot2)

            if opt.cuda:
                real_data = real_data.cuda()
                fake_data = fake_data.cuda()

            # compute grad for real data
            errD_real = netD(real_data)
            errD_real.backward(one)

            # compute grad for fake data
            errD_fake = netD(fake_data)
            errD_fake.backward(mone)

            # add gradient penalty if necessary
            if opt.do_gp:
                gradient_penalty = calc_gradient_penalty(netD, real_data.data, fake_data.data)
                gradient_penalty.backward()

            errD = errD_real - errD_fake + (0 if not opt.do_gp else gradient_penalty)
            Wasserstein_D = errD_fake - errD_real

            # print wasserstein estimate
            if i % opt.print_every_itr == 0:
                print(f'per batch W1 estimate: {-Wasserstein_D.data[0]}')

            # step with gradients
            optimizerD.step()

        # now with a fixed D, estimate distance on all data
        print(f'\nend of epoch {epoch}, computing estimate over all data...')
        lst_avg = []
        with torch.no_grad():
            for i in range(per_epoch_iters):
                if i % opt.print_every_itr == 0:
                    print(f'epoch {epoch}: {i}/{per_epoch_iters} iterations')

                # get the data
                real_data = load_batch(onlyfiles_real[i*opt.batchSize:(i+1)*opt.batchSize], opt.dataroot1)
                fake_data = load_batch(onlyfiles_fake[i*opt.batchSize:(i+1)*opt.batchSize], opt.dataroot2)

                if opt.cuda:
                    real_data = real_data.cuda()
                    fake_data = fake_data.cuda()

                # compute grad for real data
                errD_real = netD(real_data)
                errD_fake = netD(fake_data)
                errD = errD_real - errD_fake
                Wasserstein_D = -errD
                lst_avg.append(Wasserstein_D.item()) # note negative

        print(f'W1 estimate: {sum(lst_avg)/len(lst_avg)}')
        epoch_hist.append(sum(lst_avg)/len(lst_avg))

    # save the epoch hist as a plot
    plt.plot(epoch_hist)
    plt.title('W1 Estimate Convergence')
    plt.ylabel('W1 Estimate')
    plt.xlabel('Epoch')
    plt.savefig('w1_epochs_convergence.png')
    print("Saved plots.")
