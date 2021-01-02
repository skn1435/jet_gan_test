import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd


N_DIM_RANDOM = 96
SIZE_BATCH = 64
SIZE_HALF_BATCH = 32
N_EPOCH = 5000

FACTOR_TOLERANCE = 3.0
FACTOR_HYSTERESIS = 1.25
LR_D = 0.0005
LR_G = 0.0005

#NUMBER_TARGET = 8


# train and test
def load_train_test() :
	mnist = keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	#x_train = x_train[y_train == num, :, :]
	#y_train = y_train[y_train == num]
	#x_test = x_test[y_test == num, :, :]
	#y_test = y_test[y_test == num]
	#print(num)
	print(x_train.shape)
	print(y_train.shape)
	#x_train = 2.0 * x_train / np.max(x_train) - 1.0
	#x_test = 2.0 * x_test / np.max(x_test) - 1.0
	x_train = x_train / np.max(x_train)
	x_test = x_test / np.max(x_test)
	x_train = x_train.reshape(len(x_train), 28, 28, 1)
	x_test = x_test.reshape(len(x_test), 28, 28, 1)
	return (x_train, y_train), (x_test, y_test)

# model
def build_generator(dim_noise = N_DIM_RANDOM, shape_image = [28, 28]) :
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(256 * np.prod(shape_image) / 16, input_dim = (dim_noise + 1), activation = "tanh"))
	#model.add(keras.layers.Dense(256 * np.prod(shape_image) / 16, input_dim = dim_noise, activation = "tanh"))
	#model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.Reshape((int(shape_image[0] / 4), int(shape_image[1] / 4), 256)))
	#model.add(keras.layers.BatchNormalization(momentum = 0.8))
	model.add(keras.layers.UpSampling2D((2, 2)))
	model.add(keras.layers.Conv2D(64, (3, 3), padding = "same", activation = "tanh"))
	#model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.UpSampling2D((2, 2)))
	model.add(keras.layers.Conv2D(1, (3, 3), padding = "same", activation = "sigmoid"))
	model.add(keras.layers.Reshape((shape_image[0], shape_image[1], 1)))
	return model

def build_discriminator(shape_image = [28, 28]) :
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(64, (3, 3), input_shape = (shape_image[0], shape_image[1], 1)))
	model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
	model.add(keras.layers.Conv2D(256, (3, 3)))
	model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
	model.add(keras.layers.Flatten())
	#model.add(keras.layers.Dense(512))
	#model.add(keras.layers.LeakyReLU(alpha = 0.2))
	model.add(keras.layers.Dense(64))
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

def write_loss(data, fname) :
	df = pd.DataFrame(data)
	df.to_csv(fname)


def main() :
	# build model
	generator = build_generator()
	#generator.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
	discriminator = build_discriminator()
	discriminator.compile(optimizer = keras.optimizers.Adam(lr = LR_D), loss = "binary_crossentropy", metrics = ["accuracy"])
	joint = build_joint(generator, discriminator)
	joint.compile(optimizer = keras.optimizers.Adam(lr = LR_G), loss = "binary_crossentropy", metrics = ["accuracy"])
	# loss
	loss_discr_gen = 0.0
	loss_discr_real = 0.0
	loss_discr = 0.0
	loss_gener = 0.0
	a_loss_discr = np.array([])
	a_loss_gener = np.array([])
	# flag
	is_bonus_discr = False
	is_bonus_gener = False
	# train
	(x_train, y_train), (x_test, y_test) = load_train_test()
	for i_epoch in range(N_EPOCH) :
		# train discriminator
		# generate train data
		num = np.random.randint(0, 10)
		xx_noise = np.random.normal(0.0, 1.0, (SIZE_HALF_BATCH, N_DIM_RANDOM))
		idx_noise = np.array([num / 4.5 - 1.0] * SIZE_HALF_BATCH)
		seed_noise = np.concatenate([xx_noise, np.array([idx_noise]).T], 1)
		x_train_gen = generator.predict(seed_noise)
		idx_train_real = np.random.randint(0, len(x_train[y_train == num]), SIZE_HALF_BATCH)
		x_train_real = (x_train[y_train == num])[idx_train_real]
		# train
		if is_bonus_gener :
			loss_discr_gen = np.array( discriminator.evaluate(x_train_gen, np.zeros((SIZE_HALF_BATCH, 1))) )
			loss_discr_real = np.array( discriminator.evaluate(x_train_real, np.ones((SIZE_HALF_BATCH, 1))) )
		else :
			loss_discr_gen = np.array( discriminator.train_on_batch(x_train_gen, np.zeros((SIZE_HALF_BATCH, 1))) )
			loss_discr_real = np.array( discriminator.train_on_batch(x_train_real, np.ones((SIZE_HALF_BATCH, 1))) )
		loss_discr = 0.5 * (loss_discr_gen + loss_discr_real)
		# train generator
		xx_noise_2 = np.random.normal(0.0, 1.0, (SIZE_BATCH, N_DIM_RANDOM))
		idx_noise_2 = np.random.randint(0, 10, SIZE_BATCH) / 4.5 - 1.0
		seed_noise_2 = np.concatenate([xx_noise_2, np.array([idx_noise_2]).T], 1)
		y_train_real = np.array([1] * SIZE_BATCH)
		if is_bonus_discr :
			loss_gener = joint.evaluate(seed_noise_2, y_train_real)
		else :
			loss_gener = joint.train_on_batch(seed_noise_2, y_train_real)
		# print
		print("%d / %d : Discr %f, Gener %f" % (i_epoch + 1, N_EPOCH, loss_discr[0], loss_gener[0]))
		a_loss_discr = np.append(a_loss_discr, loss_discr[0])
		a_loss_gener = np.append(a_loss_gener, loss_gener[0])
		# bonus time flag
		if (FACTOR_TOLERANCE * loss_discr[0] < loss_gener[0]) and (not is_bonus_gener) :
			is_bonus_gener = True
			is_bonus_discr = False
			print("--- BONUS TIME FOR Gener ---")
		elif (FACTOR_HYSTERESIS * loss_gener[0] < loss_discr[0]) and is_bonus_gener :
			is_bonus_gener = False
			is_bonus_discr = False
		elif (FACTOR_TOLERANCE * loss_gener[0] < loss_discr[0]) and (not is_bonus_discr) :
			is_bonus_gener = False
			is_bonus_discr = True
			print("--- BONUS TIME FOR Discr ---")
		elif (FACTOR_HYSTERESIS * loss_discr[0] < loss_gener[0]) and is_bonus_discr :
			is_bonus_gener = False
			is_bonus_discr = False
		#else :
		#	is_bonus_gener = False
		#	is_bonus_discr = False
		# save image
		if (i_epoch + 1) % 100 == 0 :
			xx_noise_3 = np.random.normal(0.0, 1.0, (16, N_DIM_RANDOM))
			idx_noise_3 = np.random.randint(0, 10, 16) / 4.5 - 1.0
			seed_noise_3 = np.concatenate([xx_noise_3, np.array([idx_noise_3]).T], 1)
			x_train_gen_3 = generator.predict(seed_noise_3)
			fname_out = "image_%05d_gan_number_3.csv" % (i_epoch + 1)
			write_image(x_train_gen_3, fname_out)
	# save loss
	fname_loss = "loss_gan_number_3.csv"
	write_loss([a_loss_discr, a_loss_gener], fname_loss)


if __name__ == "__main__" :
	main()
