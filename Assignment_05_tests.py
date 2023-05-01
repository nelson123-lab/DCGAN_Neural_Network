import numpy as np
import tensorflow as tf
from Joseph_05_01 import build_generator, build_discriminator, DCGAN

def test_generator():
    tf.keras.utils.set_random_seed(5368)
    generator_model = build_generator()
    assert generator_model.layers[0].input_shape == (None, 100)
    assert generator_model.layers[0].output_shape == (None, 7*7*8)
    assert generator_model.layers[1].output_shape == (None, 7*7*8)
    assert generator_model.layers[2].output_shape == (None, 7*7* 8)
    assert generator_model.layers[3].output_shape == (None, 7, 7, 8)
    assert generator_model.layers[4].output_shape == (None, 7, 7, 8)
    assert generator_model.layers[5].output_shape == (None, 7, 7, 8)
    assert generator_model.layers[6].output_shape == (None, 7, 7, 8)
    assert generator_model.layers[7].output_shape == (None, 14, 14, 16)
    assert generator_model.layers[8].output_shape == (None, 14, 14, 16)
    assert generator_model.layers[9].output_shape == (None, 14, 14, 16)
    assert generator_model.layers[10].output_shape == (None, 28, 28, 1)

    # check layer types
    assert isinstance(generator_model.layers[0], tf.keras.layers.Dense)
    assert isinstance(generator_model.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(generator_model.layers[2], tf.keras.layers.LeakyReLU)
    assert isinstance(generator_model.layers[3], tf.keras.layers.Reshape)
    assert isinstance(generator_model.layers[4], tf.keras.layers.Conv2DTranspose)
    assert isinstance(generator_model.layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(generator_model.layers[6], tf.keras.layers.LeakyReLU)
    assert isinstance(generator_model.layers[7], tf.keras.layers.Conv2DTranspose)
    assert isinstance(generator_model.layers[8], tf.keras.layers.BatchNormalization)
    assert isinstance(generator_model.layers[9], tf.keras.layers.LeakyReLU)
    assert isinstance(generator_model.layers[10], tf.keras.layers.Conv2DTranspose)

    # check output

    z = tf.random.normal([1, 100])
    before_actual = generator_model(z, training=False)
    test_vals = np.load("test_generator.npz")
    before_target = test_vals['before_actual']
    np.testing.assert_allclose(before_target, before_actual, atol=1e-5)



def test_discriminator():
    tf.keras.utils.set_random_seed(5368)
    discriminator_model = build_discriminator()
    discriminator_model.summary()
    assert discriminator_model.layers[0].input_shape == (None, 28, 28, 1)
    assert discriminator_model.layers[0].output_shape == (None, 14, 14, 16)
    assert discriminator_model.layers[1].output_shape == (None, 14, 14, 16)
    assert discriminator_model.layers[2].output_shape == (None, 14, 14, 16)
    assert discriminator_model.layers[3].output_shape == (None, 7, 7, 32)
    assert discriminator_model.layers[4].output_shape == (None, 7, 7, 32)
    assert discriminator_model.layers[5].output_shape == (None, 7, 7, 32)
    assert discriminator_model.layers[6].output_shape == (None, 1568)
    assert discriminator_model.layers[7].output_shape == (None, 1)

    # check layer types
    assert isinstance(discriminator_model.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(discriminator_model.layers[1], tf.keras.layers.LeakyReLU)
    assert isinstance(discriminator_model.layers[2], tf.keras.layers.Dropout)
    assert isinstance(discriminator_model.layers[3], tf.keras.layers.Conv2D)
    assert isinstance(discriminator_model.layers[4], tf.keras.layers.LeakyReLU)
    assert isinstance(discriminator_model.layers[5], tf.keras.layers.Dropout)
    assert isinstance(discriminator_model.layers[6], tf.keras.layers.Flatten)
    assert isinstance(discriminator_model.layers[7], tf.keras.layers.Dense)

    # generate a random image and classify it with the discriminator
    data = tf.random.normal([1, 28, 28, 1])
    before_actual = discriminator_model(data, training=False)
    assert abs(before_actual[0, 0] - (-0.30697495)) < 1e-5, "Discriminator output is incorrect on input"

def test_dcgan_train_step():
    tf.keras.utils.set_random_seed(5368)
    generator = build_generator()
    discriminator = build_discriminator()
    dcgan = DCGAN(discriminator=discriminator, generator=generator)
    dcgan.compile(d_optimizer=tf.keras.optimizers.Adam(1e-4),
                    g_optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    data = tf.random.normal([4, 28, 28, 1])

    # random latents
    z = tf.random.normal([1, 100])

    before_actual = generator(z, training=False)

    values = np.load("test_dcgan_train_step.npz")
    before_target = values['before_training']
    after_target = values['after_training']

    assert np.allclose(before_target, before_actual, atol=1e-3)

    history = dcgan.fit(data, epochs=1, batch_size=4)
    assert 'd_loss' in history.history and 'g_loss' in history.history


    after_actual = generator(z, training=False)
    assert np.allclose(after_target, after_actual, atol=1e-3) # Not working
    assert abs(history.history['d_loss'][0] - 0.708788275718689) < 1e-3 # Not working 
    assert abs(history.history['g_loss'][0] - 0.6812016367912292) < 1e-3 # Not working