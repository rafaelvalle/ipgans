#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example showing the evolution of the distribution of pixel values for GANs
trained on MNIST data under different conditions. Based on GAN implementations
by Jan Schlüter including
"""

from __future__ import print_function

import sys
import os
import time

from tqdm import tqdm
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import theano
import theano.tensor as T

import lasagne


# ################## Download and prepare the MNIST dataset ##################
def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def binarize_data(data, min_thresh, max_thresh, min_val=0, max_val=1):
    data[data < min_thresh] = 0
    data[data > max_thresh] = 1


# ##################### Build the neural network model #######################
def BatchNorm(layer, do_batch_norm=False):
    if do_batch_norm:
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_norm
        return batch_norm(layer)
    return layer


def build_generator(input_var=None, do_batch_norm=False,
                    activation='sigmoid'):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
    from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer

    # define non-linearity at last layer
    if activation == 'sigmoid':
        from lasagne.nonlinearities import sigmoid as activation_fn
    elif activation == 'scaled_tanh':
        from lasagne.nonlinearities import ScaledTanH
        activation_fn = ScaledTanH(2/3., 1.7519)
    elif activation == 'linear':
        from lasagne.nonlinearities import linear as activation_fn
    else:
        raise Exception("{} non-linearity not supported".format(activation))

    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = BatchNorm(DenseLayer(layer, 1024), do_batch_norm)
    # project and reshape
    layer = BatchNorm(DenseLayer(layer, 128*7*7), do_batch_norm)
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    layer = BatchNorm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                                    output_size=14),
                      do_batch_norm)
    layer = Deconv2DLayer(layer, 1, 5, stride=2, crop='same', output_size=28,
                          nonlinearity=activation_fn)
    print("Generator output:", layer.output_shape)
    return layer


def build_critic(gan, input_var=None, do_batch_norm=False):
    from lasagne.layers import (InputLayer, Conv2DLayer, DenseLayer)
    from lasagne.nonlinearities import LeakyRectify
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # two convolutions
    layer = BatchNorm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                  nonlinearity=lrelu),
                      do_batch_norm)
    layer = BatchNorm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                  nonlinearity=lrelu),
                      do_batch_norm)
    # fully-connected layer
    layer = BatchNorm(DenseLayer(layer, 1024, nonlinearity=lrelu),
                      do_batch_norm)
    # output layer
    if gan in ('wgan', 'wgan-gp'):
        layer = DenseLayer(layer, 1, nonlinearity=None, b=None)
    elif gan in ('lsgan', ):
        layer = DenseLayer(layer, 1, nonlinearity=None)
    elif gan in ('dcgan', ):
        from lasagne.nonlinearities import sigmoid
        layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
    else:
        raise Exception("GAN {} is not supported".format(gan))

    print("Critic output: ", layer.output_shape)
    return layer


# ############################# Batch iterator ###############################
def iterate_minibatches(inputs, targets, batch_size, shuffle=False,
                        forever=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]
        if not forever:
            break


#############################################################################
# Define number of critic runs given GAN and generator updates
def get_critic_runs(gan, generator_updates):
    if gan in ('dcgan', 'lsgan'):
        return 1

    # In each epoch, we do `epochsize` generator updates. Usually, the
    # critic is updated 5 times before every generator update. For the
    # first 25 generator updates and every 500 generator updates, the
    # critic is updated 100 times instead, following the authors' code.
    if (generator_updates < 25) or (generator_updates % 500 == 0):
        return 100
    else:
        return 5


##############################################################################
# Define method for plotting fake images and histograms
def plot_samples(gan, samples, savepath):
    # we plot 42 of them
    plt.imsave(savepath,
               (samples[:42].reshape(6, 7, 28, 28)
                            .transpose(0, 2, 1, 3)
                            .reshape(6*28, 7*28)),
               cmap='gray')
    plt.close('all')


def plot_histogram(gan, fake_samples, real_samples, title, savepath):
    # we compute histograms of pixel intensities
    fake_flatten = fake_samples.flatten()
    real_flatten = real_samples.flatten()
    fake_interval = fake_flatten[
        np.array(fake_flatten > 0) | np.array(fake_flatten < 1)]
    real_interval = real_flatten[
        np.array(real_flatten > 0) | np.array(real_flatten < 1)]
    t_min, t_max = 0.02, 0.98
    fig, axes = plt.subplots(2, 2, figsize=(8, 4))
    axes = axes.flatten()
    axes[0].hist(fake_flatten, bins=201, range=(-1.0, 1.0))
    axes[1].hist(real_flatten, bins=201, range=(-1.0, 1.0))
    axes[2].hist(fake_interval, bins=100, range=(t_min, t_max))
    axes[3].hist(real_interval, bins=100, range=(t_min, t_max))
    axes[0].set_title('fake')
    axes[1].set_title('real')
    axes[2].set_title('fake ({}, {}) interval'.format(t_min, t_max))
    axes[3].set_title('real ({}, {}) interval'.format(t_min, t_max))

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close('all')


# ############################## Main program ################################
def main(gan, optimizer, do_batch_norm, n_epochs, epoch_size, batch_size,
         initial_eta, eta_decay, threshold, activation, dump):

    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    if threshold != 0.0:
        X_train[X_train >= threshold] = 1
        X_train[X_train < threshold] = 0
        X_test[X_test >= threshold] = 1
        X_test[X_test < threshold] = 0

    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.normal((batch_size, 100), avg=0.5, std=0.1)

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var, do_batch_norm, activation)
    critic = build_critic(gan, input_var, do_batch_norm)

    # Create expression for passing real data through the critic
    fake_in = lasagne.layers.get_output(generator)
    real_out = lasagne.layers.get_output(critic)
    # Create expression for passing fake data through the critic
    fake_out = lasagne.layers.get_output(critic, fake_in)

    # Create loss expressions
    if gan == 'dcgan':
        # Create loss expressions
        generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1)
        generator_loss = generator_loss.mean()
        critic_loss = (lasagne.objectives.binary_crossentropy(real_out, 1) +
                       lasagne.objectives.binary_crossentropy(fake_out, 0))
        critic_loss = critic_loss.mean()
    elif gan == 'lsgan':
        # a, b, c = -1, 1, 0  # Equation (8) in the paper
        a, b, c = 0, 1, 1  # Equation (9) in the paper
        generator_loss = lasagne.objectives.squared_error(fake_out, c).mean()
        critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
                       lasagne.objectives.squared_error(fake_out, a).mean())
    elif gan in ('wgan', 'wgan-gp'):
        # original in Jan's code
        # generator_loss = fake_out.mean()
        # critic_loss = real_out.mean() - fake_out.mean()
        generator_loss = -fake_out.mean()
        critic_loss = -real_out.mean() + fake_out.mean()
        if gan == 'wgan-gp':
            # gradient penalty
            alpha = srng.uniform((batch_size, 1, 1, 1), low=0., high=1.)
            differences = fake_in - input_var
            interpolates = input_var + (alpha*differences)
            inter_out = lasagne.layers.get_output(critic, interpolates)
            gradients = theano.grad(inter_out.sum(), wrt=interpolates)
            slopes = T.sqrt(T.sum(T.sqr(gradients), axis=(1, 2, 3)))
            critic_penalty = 10 * T.mean((slopes-1.)**2)
            # original in Jan's code
            # critic_loss -= critic_penalty
            critic_loss += critic_penalty
    else:
        raise Exception("GAN {} is not supported".format(gan))

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))

    # choose the optimizer
    if optimizer == 'adam':
        generator_updates = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=eta, beta1=0.5,
            beta2=0.9)
        critic_updates = lasagne.updates.adam(
            critic_loss, critic_params, learning_rate=eta, beta1=0.5,
            beta2=0.9)
    elif optimizer == 'rmsprop':
        generator_updates = lasagne.updates.rmsprop(
                generator_loss, generator_params, learning_rate=eta)
        critic_updates = lasagne.updates.rmsprop(
                critic_loss, critic_params, learning_rate=eta)

    # Compile functions performing a training step on a mini-batch (according
    # to the updates dictionary) and returning the corresponding loss:
    generator_train_fn = theano.function([], generator_loss,
                                         givens={noise_var: noise},
                                         updates=generator_updates)
    critic_train_fn = theano.function([input_var], critic_loss,
                                      givens={noise_var: noise},
                                      updates=critic_updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Finally, launch the training loop.
    print("Starting training...")
    # We create an infinite supply of batches (as an iterable generator):
    batches = iterate_minibatches(X_train, y_train, batch_size, shuffle=True,
                                  forever=True)
    # build preffix and suffix str for saving files
    prefix = "{}_mnist".format(gan)
    suffix = "non_lin_{}_opt_{}_bn_{}_etadecay_{}_thresh_{}".format(
        activation, optimizer, do_batch_norm, eta_decay, threshold)

    # We iterate over epochs:
    n_generator_updates = 0
    for epoch in tqdm(range(n_epochs)):
        # sample a batch of samples, plot them inc. histograms
        n_samples = 1000
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(n_samples, 100)))
        plot_samples(gan, samples, "samples/{}_samples_{}_{}.png".format(
            prefix, epoch, suffix))
        plot_histogram(
            gan, samples, X_train, "{} : {} {}".format(gan, optimizer, epoch),
            "histograms/{}_hist_epoch_{}_{}.png".format(prefix, epoch, suffix))

        critic_scores = []
        generator_scores = []
        for _ in range(epoch_size):
            for _ in range(get_critic_runs(gan, n_generator_updates)):
                batch = next(batches)
                inputs, targets = batch
                critic_scores.append(critic_train_fn(inputs))
            generator_scores.append(generator_train_fn())
            n_generator_updates += 1

        print("  generator loss:\t\t{}".format(np.mean(generator_scores)))
        print("  critic loss:\t\t{}".format(np.mean(critic_scores)))

        # After half the epochs, we start decaying the learn rate towards zero
        if eta_decay and epoch >= int(n_epochs / 2):
            progress = float(epoch) / n_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    # dump the network weights to a file:
    if dump:
        np.savez('models/{}_mnist_gen.npz'.format(gan),
                 *lasagne.layers.get_all_param_values(generator))
        np.savez('models/{}_mnist_crit.npz'.format(gan),
                 *lasagne.layers.get_all_param_values(critic))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="""Train GANs on MNIST, generating samples and
                               histograms""",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("gan", type=str, default='dcgan',
                        help="GAN: dcgan, lsgan, wgan, wgan-gp")
    parser.add_argument("--optimizer", type=str, default='adam',
                        help="Optimizer: adam, rmsprop")
    parser.add_argument("--do_batch_norm", action='store_true',
                        help="Use batch normalization on generator and critic")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--epoch_size", type=int, default=100,
                        help="Epoch size")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-Batch size")
    parser.add_argument("--initial_eta", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--activation", type=str, default='sigmoid',
                        help="Activation function")
    parser.add_argument("--eta_decay", action='store_true',
                        help="Learning rate decay")
    parser.add_argument("--threshold", type=float, default=0, help="Dump model")
    parser.add_argument("--dump", action='store_true', help="Dump model")
    args = parser.parse_args()

    main(**vars(args))
