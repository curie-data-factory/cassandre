########################################################## READ ME ##########################################################
#                                                                                                                           #
#                   				Main script for running & evaluating predictions   										#
#                       	  The user can choose the model and make prediction on test data.                               #
#                                                                                                                           #
#############################################################################################################################

########################## Imports ##########################


from __future__ import print_function

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import layer_utils,to_categorical,plot_model
from keras import backend as K
from keras.engine.input_layer import Input 

from sklearn.metrics import r2_score

import numpy as np
import warnings,os,argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import pandas as pd
import SimpleITK as sitk
import math as m
import seaborn as sns
import random as rd
import warnings

from dltk.io.preprocessing import whitening,normalise_zero_one

from segmentation_models import *
from utils import *
from custom_generator import *
from test_model import * 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

########################## Parser & choice functions ##########################


def get_parser_pred():

	#we code the parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dimension', help = 'Type: bool. Choose the segmentation mode, whether it is 3D or 2D segmentation. Default (if this argument is not given) will be 2D', 
		action = 'store_true')
	parser.add_argument('-m', '--model', help = 'Type: choice, default AE. Choose the model you want to use between AE (auto encoder), PSPnet, UNet, VGG16, Xception and SegNet', 
		choices = ['AE','PSPnet','UNet','VGG16','Xception','SegNet'], default = 'AE')
	parser.add_argument('-o', '--operator', help = 'Type: int (Choice 0, 1 or 2), default 1. Choose the operator for ground truth dataset. If set to 0 it chooses the Y with no threshold',
		type = int, default = 1)
	parser.add_argument('-c', '--cut_axis', help = 'Type: int (Choice 0, 1, or 2), default 2 (axial). Only for 2D segmentation. Choose 2 for axial or 0 for sagittal cut',
		type = int, default = 2)
	parser.add_argument('-w', '--weighted_map', help = 'Type: bool. If set to true then Y vector are transformed into weighted map instead of one hot vectors',
		action = 'store_true')
	parser.add_argument('-f','--function_metric',  help = 'Type: str. Choose the metric compiled with the model',
		choices = ['acc','Dice','SDice','F1','MC'], default = 'acc')
	parser.add_argument('-p', '--plot_all', help = 'Type: bool. If set to true then plot also background predictions',
		action = 'store_true')
	parser.add_argument('-t', '--threshold', help = 'Type: float (range 0 to 1), default 0. If weighted map, then apply a mask on predictions. If negative, then no mask is applied',
		type = float, default = 0)

	return parser


def choose_metric(function_metric):

    if function_metric == 'Dice':
        return Dice
    elif function_metric == 'SDice':
        return Sorensen_Dice
    elif function_metric == 'MC':
    	return matthews_correlation
    else: 
        return F1_score


def choose_metric_from_arg(function_metric):

	if function_metric == 'SDice':
		return 'Sorensen_Dice'
	elif function_metric == 'MC':
		return 'matthews_correlation'
	elif function_metric == 'F1': 
		return 'F1_score'

########################## Main ##########################


if __name__ == '__main__':

	# We ignore warnings
	warnings.filterwarnings('ignore')

	# We retrieve the arguments (we use the same parser as the one for test_model.py)
	arg = get_parser_pred().parse_args()

	dimension = arg.dimension
	function_metric = arg.function_metric
	model_str = arg.model
	operator = arg.operator
	cut_axis = arg.cut_axis
	weighted_map = arg.weighted_map
	threshold = arg.threshold
	

	try:
		name_desktop = os.environ['COMPUTERNAME']
		if name_desktop == 'ICH03626':
			bug = True 
	except KeyError:
		bug = False

	metric_str = choose_metric_from_arg(function_metric)

	# Now we load data for generator via the sampling info list 
	cut_name = name_cut(cut_axis) 
	sampling_info_path = choosing_savename(bug,dimension,'Sampling')
	sampling_info_path = sampling_info_path + '_op_' + str(operator) + cut_name + '.npz'
	
	sampling_data = np.load(sampling_info_path)['a']
	patient_for_pred = sampling_data[2][0]
	full_path = sampling_data[2][1]

	print('Patients for predictions : {0} / path to prediction : {1} '.format(patient_for_pred,full_path))
	# datagen_test = DataGenerator(sampling_data, 'test', cut_axis, operator, all_image)

	with tf.Session(config=tf.ConfigProto(device_count={ "CPU": 24 },inter_op_parallelism_threads=24,intra_op_parallelism_threads=24)) as sess:
		
		# Setting session, plot data and NN model 
		K.tensorflow_backend.set_session(sess)
		
		# Loading model
		model_path = '../models/{}_{}.h5'.format(model_str,metric_str)
		if function_metric == 'acc':
			model = load_model(model_path)
		else: 
			# model = load_model(model_path,custom_objects={metric_str : choose_metric(function_metric), 'loss':weighted_categorical_crossentropy()})
			model = load_model(model_path,custom_objects={metric_str : choose_metric(function_metric)})
		
		print('Choosen model : {0} with metric {1} '.format(model_str,metric_str))
		
		# Running predicitons 
		# pred = model.predict_generator(generator=datagen_test,verbose=1,use_multiprocessing=True)
		# plot_model(model, to_file='./figure/{0}_{1}.png'.format(model_str,function_metric))

		list_X, list_Y = [],[]
		for patients in range(patient_for_pred): 
			patient_name = 'patient_' + str(patients+1) + '_op_' + str(operator) + cut_name + '.npz'
			name_X = os.path.join(full_path, 'MRI', patient_name)
			name_Y = os.path.join(full_path, 'Segmentation', patient_name)
			print(name_X,name_Y)

			X = np.load(name_X)['a']
			Y = np.load(name_Y)['a']
			X = X_train_transform(X)

			list_X.append(X)
			list_Y.append(Y)

		#X_new = np.concatenate(list_X,axis=0)
		#Y_new = np.concatenate(list_Y,axis=0)
		random_patient = rd.randint(0,7)
		X_new = list_X[random_patient]
		Y_new = list_Y[random_patient]
		print('Random patient choosen: {}'.format(random_patient+1))

		pred = model.predict(X_new, verbose=1)
		print('\nX shape : {0} Y shape : {1} pred shape : {2}'.format(X_new.shape,Y_new.shape,pred.shape))
		
		# Reshaping data for vizualisation 
		X_new = np.reshape(X_new,(-1,416,416))

		if weighted_map: 
			Y_new = Y_new[:,:,:,0]
			pred = pred[:,:,:,0]

		else: 
			Y_new = Y_new[:,:,:,1]
			pred = pred[:,:,:,1]

		# Choosing random samples 
		sample_size = 3
		mini = pred.shape[0]
		list_of_index = rd.sample(range(0,mini),sample_size)
		print('Choosen Sample : {}\n'.format(list_of_index))
		plt.figure(figsize=(3*sample_size, 3*sample_size))
		dice_list = []

		# Plot loop 
		for i,sample in enumerate(list_of_index):

			current_pred = get_frame(0,pred,sample)
			current_X = get_frame(0,X_new,sample)
			current_truth = get_frame(0,Y_new,sample)

			if weighted_map and threshold > 0 and threshold < 1: 
				
				print('Creating mask keeping the top {} % values'.format(100*threshold))
				p = np.quantile(current_pred,1-threshod)
				binary_pred = [current_pred >= p] 
				current_pred = binary_pred
				current_pred = np.asarray(current_pred)
				current_pred = current_pred[0,:,:]

			plt.subplot(3,sample_size,i+1)
			plt.imshow(current_X.T, cmap="gray", origin="lower")
			plt.subplot(3,sample_size,i+1+sample_size)
			plt.imshow(current_pred.T, cmap="gray", origin="lower")
			plt.subplot(3,sample_size,i+1+2*sample_size)
			plt.imshow(current_truth.T, cmap="gray", origin="lower")
			
			dice = Sorensen_Dice(current_truth,current_pred)
			dice_list.append(dice.eval())
		
		plt.suptitle("Prediction on test data")
		plt.savefig('../figure/plot_preds_for_{0}_{1}.png'.format(model_str,metric_str))
		print(dice_list)