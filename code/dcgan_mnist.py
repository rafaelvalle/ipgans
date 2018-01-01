#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example employing Lasagne for digit generation using the MNIST dataset and
Deep Convolutional Generative Adversarial Networks
(DCGANs, see http://arxiv.org/abs/1511.06434).

It is based on the MNIST example in Lasagne:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

Note: In contrast to the original paper, this trains the generator and
discriminator at once, not alternatingly. It's easy to change, though.

Jan Schlüter, 2015-12-16
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from tqdm import tqdm


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

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


# ##################### Build the neural network model #######################
# We create two models: The generator and the discriminator network. The
# generator needs a transposed convolution layer defined first.

def build_generator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
    try:
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    except ImportError:
        raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                          "version: http://lasagne.readthedocs.io/en/latest/"
                          "user/installation.html#bleeding-edge-version")
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    # from lasagne.nonlinearities import sigmoid
    from lasagne.nonlinearities import ScaledTanH
    scaled_tanh = ScaledTanH(2/3., 1.7519)
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*7*7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                                     output_size=14))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, crop='same', output_size=28,
                          nonlinearity=scaled_tanh)
    print("Generator output:", layer.output_shape)
    return layer


def build_discriminator(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, DenseLayer)
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer
    layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
    print("Discriminator output:", layer.output_shape)
    return layer


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False,
                        forever=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
        if not forever:
            break


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=100, epochsize=100, batchsize=64, initial_eta=1e-4,
         optimizer='adam'):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var)
    discriminator = build_discriminator(input_var)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator,
                                         lasagne.layers.get_output(generator))

    # Create loss expressions
    generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
    discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1) +
                          lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))

    if optimizer == 'adam':
        generator_updates = lasagne.updates.adam(
                generator_loss, generator_params, learning_rate=eta, beta1=0.5)
        discriminator_updates = lasagne.updates.adam(
                discriminator_loss, discriminator_params, learning_rate=eta, beta1=0.5)
    elif optimizer == 'rmsprop':
        generator_updates = lasagne.updates.rmsprop(
                generator_loss, generator_params, learning_rate=eta)
        discriminator_updates = lasagne.updates.rmsprop(
                discriminator_loss, discriminator_params, learning_rate=eta)
    else:
        raise Exception("Optimizer {} not supported".format(optimizer))

    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.uniform((batchsize, 100))

    # Compile functions performing a training step on a mini-batch (according
    # to the updates dictionary) and returning the corresponding score:
    generator_train_fn = theano.function([], generator_loss,
                                         givens={noise_var: noise},
                                         updates=generator_updates)

    discriminator_train_fn = theano.function([input_var], discriminator_loss,
                                             givens={noise_var: noise},
                                             updates=discriminator_updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Finally, launch the training loop.
    print("Starting training...")
    # We create an infinite supply of batches (as an iterable generator):
    batches = iterate_minibatches(X_train, y_train, batchsize, shuffle=True,
                                  forever=True)
    # We iterate over epochs:
    for epoch in range(num_epochs):
        start_time = time.time()

        # we sample a large batch of samples
        n_samples = 1000
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(n_samples, 100)))

        # we plot 42 of them
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            plt.imsave('dcgan_mnist_samples.png',
                       (samples[:42].reshape(6, 7, 28, 28)
                                    .transpose(0, 2, 1, 3)
                                    .reshape(6*28, 7*28)),
                       cmap='gray')

        # we compute histograms of pixel intensities
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].hist(samples.flatten(), bins=100, range=(0.0, 1.0))
        axes[1].hist(X_train.flatten(), bins=100, range=(0.0, 1.0))
        axes[0].set_title('fake')
        axes[1].set_title('real')
        fig.suptitle("DCGAN with {}".format(optimizer))
        plt.savefig("histograms/mnist_dcgan_histogram_epoch_{}".format(epoch))

        # In each epoch, we do a full pass over the training data:
        discriminator_losses = []
        generator_losses = []
        for _ in tqdm(range(epochsize)):
            inputs, targets = next(batches)
            discriminator_losses.append(discriminator_train_fn(inputs))
            generator_losses.append(generator_train_fn())

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  generator loss: {}".format(np.mean(generator_losses)))
        print("  discriminator loss:  {}".format(np.mean(discriminator_losses)))

        # After half the epochs, we start decaying the learn rate towards zero
        # if epoch >= num_epochs // 2:
        #    progress = float(epoch) / num_epochs
        #    eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('dcgan_mnist_gen.npz', *lasagne.layers.get_all_param_values(generator))
    np.savez('dcgan_mnist_disc.npz', *lasagne.layers.get_all_param_values(discriminator))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DCGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS [EPOCHSIZE]]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 1000)")
        print("EPOCHSIZE: number of network updates per epoch (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['epochsize'] = int(sys.argv[2])
        main(**kwargs)
