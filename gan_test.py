import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd


SIZE_BATCH = 256
SIZE_HALF_BATCH = 128
N_EPOCH = 100


# train and test
def load_train_test() :
	mnist = keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	print(x_train.shape)
	print(y_train.shape)
	x_train = 2.0 * x_train / np.max(x_train) - 1.0
	x_test = 2.0 * x_test / np.max(x_test) - 1.0
	return (x_train, y_train), (x_test, y_test)

# model
def build_generator(dim_noise = 256, shape_image = [28, 28]) :
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(256, input_dim = dim_noise))
	model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.BatchNormalization(momentum = 0.8))
	model.add(keras.layers.Dense(512))
	model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.BatchNormalization(momentum = 0.8))
	model.add(keras.layers.Dense(np.prod(shape_image), activation = "sigmoid"))
	model.add(keras.layers.Reshape(shape_image))
	return model

def build_discriminator(shape_image = [28, 28]) :
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(256, input_shape = shape_image))
	model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.Dense(32))
	model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.Dense(1, activation = "sigmoid"))
	return model

def build_joint(generator, discriminator) :
	discriminator.trainable = False
	model = keras.Sequential([generator, discriminator])
	return model

def write_image(data, fname) :
	df = pd.DataFrame(data.reshape(16, 28*28))
	df.to_csv(fname)


def main() :
	# build model
	generator = build_generator()
	#generator.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
	discriminator = build_discriminator()
	discriminator.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
	joint = build_joint(generator, discriminator)
	joint.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
	# train
	(x_train, y_train), (x_test, y_test) = load_train_test()
	for i_epoch in range(N_EPOCH) :
		# train discriminator
		# generate train data
		xx_noise = np.random.normal(0.0, 1.0, (SIZE_HALF_BATCH, 256))
		x_train_gen = generator.predict(xx_noise)
		idx_train_real = np.random.randint(0, len(x_train), SIZE_HALF_BATCH)
		x_train_real = x_train[idx_train_real]
		# train
		loss_discr_gen = np.array( discriminator.train_on_batch(x_train_gen, np.ones((SIZE_HALF_BATCH, 1))) )
		loss_discr_real = np.array( discriminator.train_on_batch(x_train_real, np.zeros((SIZE_HALF_BATCH, 1))) )
		loss_discr = 0.5 * (loss_discr_gen + loss_discr_real)
		# train generator
		xx_noise_2 = np.random.normal(0.0, 1.0, (SIZE_BATCH, 256))
		y_train_real = np.array([1] * SIZE_BATCH)
		loss_gener = joint.train_on_batch(xx_noise_2, y_train_real)
		# print
		print("%d : Discr %f, Gener %f" % (i_epoch, loss_discr[0], loss_gener[0]))
		# save image
		xx_noise_3 = np.random.normal(0.0, 1.0, (16, 256))
		x_train_gen_3 = generator.predict(xx_noise_3)
		for i_image in range(16) :
			fname_out = "image_%03d_gan_test.csv" % i_epoch
			write_image(x_train_gen_3, fname_out)


if __name__ == "__main__" :
	main()
