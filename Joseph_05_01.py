




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
    """
    
    
    """
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
    """
    The discriminator model takes an image input with a shape of (28, 28, 1) and outputs a single
    value that indicates the probability of the input image being real or fake.

    Model Architecture:
    1. Conv2D layer with 16 filters, kernel size of (5, 5), strides of (2, 2), and padding set to 'same'.
    2. LeakyReLU activation function (default parameters)
    3. Dropout layer with a rate of 0.3.
    4. Conv2D layer with 32 filters, kernel size of (5, 5), strides of (2, 2), and padding set to 'same'.
    5. LeakyReLU activation function (default parameters)
    6. Dropout layer with a rate of 0.3.
    7. Flatten layer to convert the feature maps into a 1D array.
    8. Dense layer with 1 output neuron.

    Returns:
    model (tf.keras.models.Sequential): A TensorFlow Keras Sequential model representing the discriminator.
    """
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
    """
    The generator model takes a 100-dimensional noise vector as input and outputs a generated
    image with a shape of (28, 28, 1).

    Model Architecture:
    1. Dense layer with 7 * 7 * 8 (392) neurons and no bias, input shape of (100,).
    2. Batch normalization layer, default params
    3. LeakyReLU activation function with default params
    4. Reshape layer to convert the 1D array into a 3D feature map with a shape of (7, 7, 8).
    5. Conv2DTranspose (deconvolution) layer with 8 filters, kernel size of (5, 5), strides of (1, 1)
    6. Batch normalization layer.
    7. LeakyReLU activation function with default params
    8. Conv2DTranspose (deconvolution) layer with 16 filters, kernel size of (5, 5), strides of (2, 2)
    9. Batch normalization layer.
    10. LeakyReLU activation function with default params
    11. Conv2DTranspose (deconvolution) layer with 1 filter, kernel size of (5, 5), strides of (2, 2), with tanh activation included

    Note: For all Conv2DTranspose, use padding='same' and use_bias=False.

    Returns:
        model (tf.keras.models.Sequential): A TensorFlow Keras Sequential model representing the generator.
    
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


class DCGAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):

        batch_size = tf.shape(data)[0]

        # Train the discriminator.
        noise_data = tf.random.uniform(shape = (batch_size, 100))
        fake_images = self.generator(noise_data)
        combined_images = tf.concat([data, fake_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train the generator.
        labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise_data)
            predictions = self.discriminator(fake_images)
            g_loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}



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
