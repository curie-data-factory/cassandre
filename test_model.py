########################################################## READ ME ##########################################################
#                                                                                                                           #
#                   								Main script for tests            										#
#                       		The user can choose the model and custom parameters for test.                               #
#                    Note that each time you want to test with different data you have to add the -a argument               #
#                                                                                                                           #
#############################################################################################################################

########################## Imports ##########################


from __future__ import print_function

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import layer_utils,to_categorical
from keras import backend as K
from keras.engine.input_layer import Input 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

import numpy as np
import warnings,os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import pandas as pd
import SimpleITK as sitk
import math as m
import warnings
import sys 

from tqdm import tqdm
from dltk.io.augmentation import add_gaussian_noise, flip, extract_class_balanced_example_array
from dltk.io.preprocessing import whitening,normalise_zero_one

from segmentation_models import *
from utils import *
from image_processing import * 
from custom_generator import *
from custom_callback import * 


########################## Parser ##########################


def get_parser():

	#we code the parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', help = 'Type: choice, default UNet. Choose the model you want to use between AE (auto encoder), PSPnet, UNet, VGG16, Xception and SegNet', 
		choices = ['AE','PSPnet','UNet','VGG16','Xception','SegNet'], default = 'UNet')
	parser.add_argument('-d', '--dimension', help = 'Type: bool. Choose the segmentation mode, whether it is 3D or 2D segmentation. Default (if this argument is not given) will be 2D', 
		action = 'store_true')
	parser.add_argument('-s', '--size_of_minibatch', help = 'Type: int. Default: 200 in 2D segmentation, for 3D, 2. Set the sizes for minibatches training', 
		type = int, default = 200)
	parser.add_argument('-e', '--epochs', help = 'Type: int, default: 5. Choose number of epochs for the training part', 
		type = int, default = 3)
	parser.add_argument('-v', '--validation_split', help = 'Type: float between 0 and 1, default 0,1. Choose the fraction of data used during validation for each epochs',
		type = float, default = 0.1)
	parser.add_argument('-t', '--test_size', help = 'Type: float between 0 and 1, default 0,1. Choose the fraction of data used for final model evaluation',
		type = float, default = 0.1)
	parser.add_argument('-o', '--operator', help = 'Type: int (Choice 0, 1 or 2), default 1. Choose the operator for ground truth dataset. If set to 0 it chooses the Y with no threshold',
		type = int, default = 1)
	parser.add_argument('-c', '--cut_axis', help = 'Type: int (Choice 0, 1, or 2), default 2 (axial). Only for 2D segmentation. Choose 2 for axial or 0 for sagittal cut',
		type = int, default = 2)
	parser.add_argument('-a', '--add', help	= 'Type: bool. Run an additional function that creates and save a dataset from /var/www/data. Use when adding data in the /data folder',
		action = 'store_true')
	parser.add_argument('-p', '--padding', help	= 'Type: int. Choose the padding for X dataset creation. Default 5',
		type = int, default = 5)
	parser.add_argument('-w', '--weighted_map', help = 'Type: bool. If set to true then Y vector are transformed into weighted map instead of one hot vectors',
		action = 'store_true')
	parser.add_argument('-f','--finetune', help = 'Type: bool. Choose if you want to fine tune the model regarding optimizers',
		action = 'store_true')

	return parser


def check_arg_integrity(arg_list,namelist): 

	for arg,name in zip(arg_list,namelist): 
		
		# The only two float arg are proportion of test and val set. They must be below one 
		if type(arg) == float and (arg >= 1 or arg < 0):
			raise ValueError('{0} Argument requires to be between 0 and 1 (excluded). You gave a value of {1}'.format(name,arg))
		# The other arguments are int, they must be equals to 0 1 or 2 only if this is about cut_axis and operator.
		elif name in ['operator','cut_axis'] and (arg > 2 or arg < 0): 
			raise ValueError('{0} Argument must be a int equals to 0, 1 or 2. You gave a value of {1}'.format(name,arg))
		# The other arguments must be positive integers 
		elif name in ['padding','epochs'] and arg < 0: 
			raise ValueError('{0} Argument must be a int of value greater or equal to 0. You gave a value of {1} '.format(name,arg))
			


########################## Test model with default generated batch ##########################


def train_test_creation(validation_split,size_of_minibatch,dimension,model,bug,operator,cut_axis,add,padding,weighted_map):

	print('\n==========================================================================================================================================================\n')

	# Get saved data or create the raw dataset entirely
	if add: 
		images,segmented_img = pre_processing(dimension,operator,cut_axis,bug,padding,weighted_map)
	else: 
		images,segmented_img = get_preprocessed_data(dimension,bug,operator,cut_axis)


	# Create test and training datasets
	print('Creating test and train list ...')
	X_train,y_train,X_test,y_test = split_train_test(images,segmented_img,validation_split,dimension)		

	# Transform X_train. 
	batch_size = size_of_minibatch
	X_train = X_train_transform(X_train)
	model_str = model
	input_img = get_dim_for_model(X_train,dimension)

	return X_train,y_train,X_test,y_test, input_img, model_str, batch_size


def test_one_model(epochs,test_size,validation_split,size_of_minibatch,dimension,model,bug,operator,cut_axis,add,padding,weighted_map): 

	# Get data and initialize generator 
	X_train,y_train,X_test,y_test,input_img,model_str,batch_size = train_test_creation(validation_split,size_of_minibatch,dimension,model,bug,operator,cut_axis,add,padding,weighted_map)
	datagen = ImageDataGenerator()

	# Setting session (for mutlithreading / mutliprocessing)
	with tf.Session(config=tf.ConfigProto(device_count={ "CPU": 24 },inter_op_parallelism_threads=24,intra_op_parallelism_threads=24)) as sess:
		
		K.tensorflow_backend.set_session(sess)

		classes = choose_classes(weighted_map)

		model,metric = choose_model_from_arg(model,dimension,input_img,classes)

		if bug: 
			for e in range(epochs):
				print('\n=============================================== Epoch', e+1, '===============================================\n')
				batches = 0
				for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
					
					# We get out of the loop immediately
					try:
						model.fit(x_batch, y_batch,validation_data=(X_test,y_test))	
					except MemoryError:
						break

		else: 
			# With this method, history.history will always be a dict with only one value per key, so we have to process it differently if we want to plot results 
			data_dict = get_history(metric,initialize=True)
			for e in range(epochs):
				print('\n=============================================== Epoch', e+1, '===============================================\n')
				batches = 0
				for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
					history = model.fit(x_batch, y_batch,validation_data=(X_test,y_test))
					get_history(metric,data_dict=data_dict,history=history)
					
					# Update batch count 
					batches += 1
					if batches >= len(X_train) / batch_size:
						# we need to break the loop by hand because the generator loops indefinitely
						break 
			print('\n==========================================================================================================================================================\n')
			print('Training finished. ploting results...')
			plot_model_perf(data_dict,metric,model_str)
			print('Testing model ...')
			score, acc = model.evaluate(X_test, y_test)
			print('Test accuracy:', acc)


########################## Test model with custom generated batch ##########################


def train_test_in_dir(test_size,validation_split,dimension,bug,operator,cut_axis,padding,weighted_map):

	print('\n==========================================================================================================================================================\n')

	# Get saved data or create the raw dataset entirely
	images,segmented_img = pre_processing(dimension,operator,cut_axis,bug,padding,weighted_map)
	
	# Create test and training datasets
	print('Creating train test and validation directories ...')
	split_train_test_in_dir(images,segmented_img,validation_split,test_size,cut_axis,operator,dimension,bug)


def setting_up_test(dimension,bug,cut_axis,operator,model): 

	# Load data for generator creation  
	sampling_info_path = choosing_savename(bug,dimension,'Sampling')
	cut_name = name_cut(cut_axis) 
	sampling_info_path = sampling_info_path + '_op_' + str(operator) + cut_name + '.npz'
	sampling_data = np.load(sampling_info_path)['a']
	
	# Get dimension input for NN, it is given by a random X vector shape (they all have the same anyway) we put in the sampling_data dictionnary
	X_dim = sampling_data[3][0]
	input_img = get_dim_for_model(X_dim,dimension,just_shape=True)
	
	# Create custom data generators 
	datagen_train = DataGenerator(sampling_data, 'train', cut_axis, operator)
	datagen_test = DataGenerator(sampling_data, 'test', cut_axis, operator)
	datagen_val = DataGenerator(sampling_data, 'val', cut_axis, operator)

	model_str = model

	return model_str, input_img, datagen_train, datagen_val, datagen_test


def test_training_from_dir(epochs,test_size,validation_split,size_of_minibatch,dimension,model,bug,operator,cut_axis,add,padding,finetune,weighted_map): 

	# If the est needs to have a complete different set of data we have to recreate the folders
	if add: 
		train_test_in_dir(test_size,validation_split,dimension,bug,operator,cut_axis,padding,weighted_map)
	else: 
		pass 

	# If the user wants to fine tune the optimizer & learning rate for model he will have to type its name and the lr he wants on bash 
	if finetune: 
		
		# Display info on bash
		print('\n==========================================================================================================================================================')
		string = 'Fine tune optimizer and learning rate'
		centered = string.center(142)
		print('\n',centered,'\n')
		print('==========================================================================================================================================================\n')
		
		# Checking arg integrity
		check_list = ['RMSprop','Adam','Adamax','Nadam','SGD','Adadelta','Adagrad']
		input_optimizer_name = input('Choose the optimizer in the following list {0} for {1} test run (respect caps): '.format(check_list,model))
		input_optimizer_lr = float(input('Choose the learning rate for this optimizer (must be above 0 and below 1) : '))
		try:
			# Raise an error if the user didn't type a correct value 
			if input_optimizer_name not in check_list:
				raise ValueError('You didnt choose a correct optimizer. {0} not in {1}'.format(input_optimizer_name,check_list))
			elif input_optimizer_lr < 0 or input_optimizer_lr > 1: 
				raise ValueError('You didnt choose a correct learning rate. You gave a value of {0} which is not between 0 and 1'.format(input_optimizer_lr))
		except ValueError:
			raise

		input_optimizer = [input_optimizer_name,input_optimizer_lr]
	else: 
		input_optimizer = None

	# Create data generators & dimensions variable for model 
	model_str, input_img, datagen_train, datagen_val, datagen_test = setting_up_test(dimension,bug,cut_axis,operator,model)

	with tf.Session(config=tf.ConfigProto(device_count={ "CPU": 24 },inter_op_parallelism_threads=24,intra_op_parallelism_threads=24)) as sess:
		
		# Setting session, plot data and NN model 
		K.tensorflow_backend.set_session(sess)
		classes = choose_classes(weighted_map)
		model,metric = choose_model_from_arg(model,dimension,input_img,classes,input_optimizer)		
		data_dict = get_history(metric,initialize=True)

		# Define callbacks 
		MCP = ModelCheckpoint('../models/.{0}_best.hdf5'.format(model_str), save_best_only=True, monitor='loss', mode='min')
		RLROP = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1, epsilon=1e-4, mode='min')
		HIST = Batch_history(metric,model_str)
		ES = EarlyStopping(monitor='loss', min_delta=1e-4, patience=0, verbose=0, mode='min')

		# Model training 
		if not bug: 

			history = model.fit_generator(generator=datagen_train, validation_data=datagen_val, use_multiprocessing=True, epochs=epochs, callbacks=[MCP,RLROP,HIST,ES])
			get_history(metric,data_dict=data_dict,history=history)

			# Evaluate and generate plot
			print('\n==========================================================================================================================================================\n')
			print('Training finished. ploting results...')
			plot_model_perf(history.history,metric,model_str)
			print('Testing model ...')
			score, acc = model.evaluate_generator(generator=datagen_test)
			print('Test accuracy:', acc)

			# Saving model
			model_name = '../models/{0}_{1}.h5'.format(model_str,metric)
			model.save(model_name)
			print("Saved model to disk")

		else: 
			print('Managed to create session')

			# history = model.fit_generator(generator=datagen_train, validation_data=datagen_val, use_multiprocessing=True, epochs=epochs, callbacks=[MCP,RLROP,HIST,ES])


########################## testing optimizers & hyperparameters ##########################


def test_optimizers(epochs,test_size,validation_split,dimension,model,bug,operator,cut_axis,weighted_map):

	model_str, input_img, datagen_train, datagen_val, datagen_test = setting_up_test(dimension,bug,cut_axis,operator,model)
	more = False
	model_list = ['PSPnet','UNet']

	with tf.Session(config=tf.ConfigProto(device_count={ "CPU": 24 },inter_op_parallelism_threads=24,intra_op_parallelism_threads=24)) as sess:
		
		# Setting session, plot data and NN model 
		K.tensorflow_backend.set_session(sess)
		classes = choose_classes(weighted_map)
		optimizers_dict = {'Adam':0.001,'Nadam':0.002,'Adadelta':1.0,'Adagrad':0.01}
		
		# We test each optimizers and each model 
		for model in model_list: 
			
			model_str = model
			
			for key in optimizers_dict:
				
				learning_rate = optimizers_dict[key]
				current_optimizer = [key,learning_rate] 
				sys.stdout.write('Chosen optimizer : {0} with a learning rate of {1}\n'.format(key,learning_rate))
				model,metric = choose_model_from_arg(model_str,dimension,input_img,classes,current_optimizer,more)		

				# Define callbacks 
				MCP = ModelCheckpoint('../models/.{0}_{1}_best.hdf5'.format(model_str,key), save_best_only=True, monitor='val_{}'.format(metric), mode='max')
				RLROP = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, epsilon=1e-4, mode='min')

				# Model training for given learning rate and optimizer 
				if not bug: 
					
					history = model.fit_generator(generator=datagen_train,validation_data=datagen_val,use_multiprocessing=True,epochs=epochs,callbacks=[MCP,RLROP])
					
					# Saving model weigths & generate plot
					print('\n==========================================================================================================================================================\n')
					print('Training finished. saving results...')
					path = '../figure/data_for_' + model_str + '_' + key + '_lr_'+str(learning_rate)
					np.savez_compressed(path,a=history.history)
					
					# # Saving model
					# model_name = '../models/{0}.h5'.format(model_str)
					# model.save(model_name)
					# print("Saved model to disk")


				else: 
					print('Managed to create session') 


########################## Main ##########################


if __name__ == '__main__':

	# We ignore warnings
	warnings.filterwarnings('ignore')
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# We retrieve the arguments 
	arg = get_parser().parse_args()

	epochs = arg.epochs
	test_size = arg.test_size
	validation_split = arg.validation_split
	size_of_minibatch = arg.size_of_minibatch
	dimension = arg.dimension
	model = arg.model
	operator = arg.operator
	cut_axis = arg.cut_axis
	add = arg.add
	padding = arg.padding
	finetune = arg.finetune
	weighted_map = arg.weighted_map

	try:
		name_desktop = os.environ['COMPUTERNAME']
		if name_desktop == 'ICH03626':
			bug = True 
	except KeyError:
		bug = False


	# Checking for correct argument values 
	arg_list = [test_size,validation_split,operator,cut_axis,padding,epochs]
	namelist = ['test_size','validation_split','operator','cut_axis','padding','epochs']
	check_arg_integrity(arg_list,namelist)
	
	# testing model 
	test_training_from_dir(epochs,test_size,validation_split,size_of_minibatch,dimension,model,bug,operator,cut_axis,add,padding,finetune,weighted_map)
	# test_optimizers(epochs,test_size,validation_split,dimension,model,bug,operator,cut_axis,weighted_map)