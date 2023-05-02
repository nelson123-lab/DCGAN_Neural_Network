# Joseph, Nelson
# 1002_050_500
# 2023_05_01
# Assignment_05_01

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_images(generated_images, n_rows=1, n_cols=10):
    """
    Plot the images in a 1x10 grid
    :param generated_images:
    :return:
    """
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    ax = ax.flatten()
    for i in range(n_rows*n_cols):
        ax[i].imshow(generated_images[i, :, :], cmap='gray')
        ax[i].axis('off')
    return f, ax

class GenerateSamplesCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, noise):
        self.generator = generator
        self.noise = noise

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists("generated_images"):
            os.mkdir("generated_images")
        generated_images = self.generator(self.noise, training=False)
        generated_images = generated_images.numpy()
        generated_images = generated_images*127.5 + 127.5
        generated_images = generated_images.reshape((10, 28, 28))
        # plot images using matplotlib
        plot_images(generated_images)
        plt.savefig(os.path.join("generated_images", f"generated_images_{epoch}.png"))
        # close the plot to free up memory
        plt.close()

def build_discriminator():
    # Model Architecture
    model = tf.keras.models.Sequential()
    # Conv2D layer with 16 filters, kernel size of (5, 5), strides of (2, 2), and padding set to 'same'.
    model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = (5, 5), strides = (2, 2), padding = 'same', input_shape = (28, 28, 1)))
    # LeakyReLU activation function (default parameters)
    model.add(tf.keras.layers.LeakyReLU())
    # Dropout layer with a rate of 0.3.
    model.add(tf.keras.layers.Dropout(rate = 0.3))
    # Conv2D layer with 32 filters, kernel size of (5, 5), strides of (2, 2), and padding set to 'same'.
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), strides = (2, 2), padding = 'same'))
    # LeakyReLU activation function (default parameters)
    model.add(tf.keras.layers.LeakyReLU())
    # Dropout layer with a rate of 0.3.
    model.add(tf.keras.layers.Dropout(rate = 0.3))
    # Flatten layer to convert the feature maps into a 1D array.
    model.add(tf.keras.layers.Flatten())
    # Dense layer with 1 output neuron.
    model.add(tf.keras.layers.Dense(units = 1))

    return model


def build_generator():
    # Model Architecture
    model = tf.keras.models.Sequential()
    # Dense layer with 7 * 7 * 8 (392) neurons and no bias, input shape of (100,).
    model.add(tf.keras.layers.Dense(units = 7*7*8, input_shape = (100,), use_bias = False))
    # Batch normalization layer, default params
    model.add(tf.keras.layers.BatchNormalization())
    # LeakyReLU activation function with default params
    model.add(tf.keras.layers.LeakyReLU())
    # Reshape layer to convert the 1D array into a 3D feature map with a shape of (7, 7, 8).
    model.add(tf.keras.layers.Reshape(target_shape = (7, 7, 8)))
    # Conv2DTranspose (deconvolution) layer with 8 filters, kernel size of (5, 5), strides of (1, 1)
    model.add(tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size = (5, 5), strides = (1, 1), padding = 'same', use_bias = False))
    # Batch normalization layer.
    model.add(tf.keras.layers.BatchNormalization())
    # LeakyReLU activation function with default params
    model.add(tf.keras.layers.LeakyReLU())
    # Conv2DTranspose (deconvolution) layer with 16 filters, kernel size of (5, 5), strides of (2, 2)
    model.add(tf.keras.layers.Conv2DTranspose(filters = 16, kernel_size = (5, 5), strides = (2, 2), padding = 'same', use_bias = False))
    # Batch normalization layer.
    model.add(tf.keras.layers.BatchNormalization())
    # LeakyReLU activation function with default params
    model.add(tf.keras.layers.LeakyReLU())
    # Conv2DTranspose (deconvolution) layer with 1 filter, kernel size of (5, 5), strides of (2, 2), with tanh activation included
    model.add(tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (5, 5), strides = (2, 2), padding = 'same', activation = 'tanh', use_bias = False))
    
    return model

# The DCGAN Class that inherits from tf.keras.Model
class DCGAN(tf.keras.Model):
    # Initializing the discriminator and the generator model.
    def __init__(self, discriminator, generator):
        # Calling the constructor of the parent class to initialize the object.
        super(DCGAN, self).__init__()
        # setting discriminator attribute to discriminator model.
        self.discriminator = discriminator
        # Setting generator attribute to generator model.
        self.generator = generator

    # Defining the compile method for the traning the DCGAN.
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        # Calling the constructer of the parent class to initialize the object.
        super(DCGAN, self).compile()
        # Setting d_optimizer attribute to discriminator optimizer.
        self.d_optimizer = d_optimizer
        # Setting g_optimizer attribute to generator optimizer.
        self.g_optimizer = g_optimizer
        # Setting loss_fn attribute to a loss function.
        self.loss_fn = loss_fn
    
    # Code referred from TensorFlow documentation.
    # Generator loss function.
    def generator_loss(self, Fake_images):
        # Creating a tensor of ones with the same shape as the Fake_images.
        Fake_labels = tf.ones_like(Fake_images)
        # Finding the loss by comparing the fake labels and fake images. 
        g_loss_value = self.loss_fn(Fake_labels, Fake_images)
        # Returns the Generator loss values of the batch of data.
        return g_loss_value
    
    # Discriminator loss function.
    def discriminator_loss(self, Real_images, Fake_images):
        # Creating tensors like ones and zeros with the same shape as Real_images and Fake_images respectively.
        Real_labels, Fake_labels = tf.ones_like(Real_images), tf.zeros_like(Fake_images)
        # Combining the real and Fake labels tensors together.
        Combined_labels = tf.concat([Real_labels, Fake_labels], axis = 0)
        # Combining the Real and Fake images into a batch for calculating the loss of the discriminator.
        Combined_outputs = tf.concat([Real_images, Fake_images], axis = 0)
        # Finding the discriminator loss shows how well the discriminator is able to distinguish between the real and fake images.
        d_loss_value = self.loss_fn(Combined_labels, Combined_outputs)
        # Returning the discriminator loss value of the batch of data.
        return d_loss_value

    # Training step in GAN algorithm.
    def train_step(self, data):
        # Calculating the input batch size of the data.
        input_batch_size = tf.shape(data)[0]
        # Generating uniform random noise with the shape of (input batch size , 100).
        Noise = tf.random.uniform([input_batch_size, 100])

        # Creating two Gradient tapes one for discriminator and one for generater for finding the generator and discriminator gradients.
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            # Generating Fake images using the generator model using the random noise.
            Fake_images = self.generator(Noise, training = True)
            # Calculates the output of discriminator model for Real Images.
            Real_output = self.discriminator(data, training = True)
            # Calculates the output of discriminator model for Fake Images.
            Fake_output = self.discriminator(Fake_images, training = True)
            # Finding the generator loss and discrimiator loss using Fake output and and combination of real and fake output respectively.
            g_loss_value, d_loss_value = self.generator_loss(Fake_output), self.discriminator_loss(Real_output, Fake_output)

        # Gradient Calculation
        # Calculating the gradients of the generator loss with respect to the trainable variables in the generator model.
        g_gradients = g_tape.gradient(g_loss_value, self.generator.trainable_variables)
        # Calculating the gradients of the discriminator loss with respect to the trainable variables in the discriminator model.
        d_gradients = d_tape.gradient(d_loss_value, self.discriminator.trainable_variables)

        # Applying the gradients to the generator model.
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        # Applying the gradients to the discriminator model.
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # Returning the Discriminator loss value and generator loss value.
        return {'discriminator_loss':d_loss_value, 'generator_loss':g_loss_value}
    
def train_dcgan_mnist():
    tf.keras.utils.set_random_seed(5368)
    # load mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # the images are in the range [0, 255], we need to rescale them to [-1, 1]
    x_train = (x_train - 127.5) / 127.5
    x_train = x_train[..., tf.newaxis].astype(np.float32)

    # plot 10 random images
    example_images = x_train[:10]*127.5 + 127.5
    plot_images(example_images)

    plt.savefig("real_images.png")


    # build the discriminator and the generator
    discriminator = build_discriminator()
    generator = build_generator()


    # build the DCGAN
    dcgan = DCGAN(discriminator=discriminator, generator=generator)

    # compile the DCGAN
    dcgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    callbacks = [GenerateSamplesCallback(generator, tf.random.uniform([10, 100]))]
    # train the DCGAN
    dcgan.fit(x_train, epochs=50, batch_size=64, callbacks=callbacks, shuffle=True)

    # generate images
    noise = tf.random.uniform([16, 100])
    generated_images = generator(noise, training=False)
    plot_images(generated_images*127.5 + 127.5, 4, 4)
    plt.savefig("generated_images.png")

    generator.save('generator.h5')


if __name__ == "__main__":
    train_dcgan_mnist()
