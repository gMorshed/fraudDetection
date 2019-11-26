from tensorflow.keras.models import load_model
import numpy as np

# Load GAN
generator = load_model('german_generator2.h5')

latent_dim = 100 # This is the dimension of the random noise we'll use for the generator

noise = np.random.normal(0, 1, (1, latent_dim))
gen_sample = generator.predict(noise)[0]

print(gen_sample)
