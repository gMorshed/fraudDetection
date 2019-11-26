import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Reshape, Input
from tensorflow.keras import Model, Sequential
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
np.random.seed(0)
import matplotlib
import matplotlib.pyplot as plt

inputDataFrame = pd.read_csv('./german_data.csv')

# Split test train
fraud_targets = pd.Series(inputDataFrame["class"])
inputDataFrame.drop(columns=["class"], inplace=True)
fraud_features = pd.DataFrame(inputDataFrame)
x_train, x_test, y_train, y_test = train_test_split(fraud_features, fraud_targets, test_size=0.2, random_state=0)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Code taken from the GAN in class exercise
img_shape = x_train[0].shape
latent_dim = 100 # This is the dimension of the random noise we'll use for the generator
batch_size = 128
epochs = 400

# Create generator_layers which are the layers for the generator model
# The generator model should have 3 dense layers of 256, 512, and 1024 units respectively
# After each Dense layer apply a BatchNormalization with a momentum value between 0.7 and 0.9
# Finalize the layers with a final output layer followed by a Reshape layer to get it to the right size

# Think about what the input to the generator is and what the output should be

generator_layers = [Dense(256, input_shape=(latent_dim,), activation="relu"), BatchNormalization(momentum=0.8),
                    Dense(512, activation="relu"), BatchNormalization(momentum=0.8),
                    Dense(1024, activation="relu"), BatchNormalization(momentum=0.8),
                    Dense(img_shape[0])]

generator = Sequential(generator_layers)

# Create discriminator_layers which are the layers for the discriminator model
# The discriminator model should have 2 Dense layers with 512, 256 units respectively
# Add the appropriate output layer and activation function

# Think about what the input and output for a discriminator model would be

discriminator_layers = [Flatten(input_shape=img_shape), Dense(512, activation="relu"),
                        Dense(256, activation="relu"), Dense(1, activation="sigmoid")]

discriminator = Sequential(discriminator_layers)
# discriminator.summary()

discriminator.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
validity = discriminator(img)

combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer="adam")

# Rescale -1 to 1
# x_train = x_train / 127.5 - 1.
# x_train = np.expand_dims(x_train, axis=3)

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Generate a batch of new images
    gen_imgs = generator.predict(noise)

    # Train the discriminator
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    discriminator.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Train the generator (to have the discriminator label samples as valid)
    discriminator.trainable = False
    g_loss = combined.train_on_batch(noise, valid)

    if epoch % 20 == 0:
        print("Loss: " + str(g_loss))
        # r, c = 5,5
        # noise = np.random.normal(0, 1, (r*c, latent_dim))
        # gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        #
        # fig, axs = plt.subplots(r, c)
        #
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.suptitle(f"Epoch: {epoch+1}")
        # plt.show()

# Generate another "digit"
# noise = np.random.normal(0, 1, (1, latent_dim))
# gen_img = generator.predict(noise)[0]
# plt.imshow(gen_img, cmap='gray')
# plt.show()

generator.save("german_generator2.h5")
