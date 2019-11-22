import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time

def train():
	start = time.time()

	DEV = False
	argvs = sys.argv
	argc = len(argvs)

	if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
		DEV = True

	if DEV:
		epochs = 2
	else:
		epochs = 100

	train_data_path = 'G:\\Study\\Projects\\Web_App_CNN\\CNN_UI_Animal_Prediction\\Data\\Train'
	test_data_path = 'G:\\Study\\Projects\\Web_App_CNN\\CNN_UI_Animal_Prediction\\Data\\Test'

	"""Parameters used for the model"""

	img_width,img_height = 300,300
	batch_size = 32
	samples_per_epoch = 400
	validation_steps = 300
	nb_filters1 = 64
	nb_filters2 = 32
	conv1_size = 5
	conv2_size = 3
	pool_size = 3
	classes_num = 3
	lr = 0.002

	model = Sequential()
	model.add(Convolution2D(nb_filters1,conv1_size,conv1_size,border_mode="same",input_shape=(img_width,img_height,3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))

	model.add(Convolution2D(nb_filters2,conv2_size,conv2_size,border_mode="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))
	model.add(Dense(classes_num,activation="softmax"))

	model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

	train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(train_data_path,target_size=(img_height,img_width),batch_size=batch_size,class_mode='categorical')

	validation_generator = test_datagen.flow_from_directory(validation_data_path,target_size=(img_height,img_width),batch_size=batch_size,class_mode='categorical')

	"""Tensorboard log"""

	log_dir = './tf-log'
	tb_cb = callbacks.Tensorboard(log_dir=log_dir,histogram_freq=0)
	cbks = [tb_cb]

	model.fit_generator(train_generator,samples_per_epoch=samples_per_epoch,epochs=epochs,validation_data=validation_generator,callbacks=cbks,validation_steps=validation_steps)

	target_dir = './models/'
	if not os.path.exists(target_dir):
		os.mkdir(target_dir)
	model.save('./models/models.h5')
	model.save_weights('./models/weights.h5')

	#For calculation of execution time
	end = time.time()
	dur = end-start

	if dur<60:
		print("Execution Time:",dur,"seconds")
	elif dur>60 and dur<3600:
		dur = dur/60
		print("Execution Time:",dur,"minutes")
	else:
		dur=dur/(60*60)
		print("Execution Time",dur,"hours")

if __name__ == "__train__":
	train()


