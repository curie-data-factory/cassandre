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

from tqdm import tqdm

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

########################## Parser ##########################


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
	parser.add_argument('-e', '--evaluate_metric', help = 'Type: choice. Choose the function to evaluate prediction on test data',
		choices = ['Dice','matthews_correlation'], default = 'Dice')
	parser.add_argument('-f','--function_metric',  help = 'Type: str. Choose the metric compiled with the model',
		choices = ['acc','Dice','Sorensen_Dice','F1_score','matthews_correlation'], default = 'acc')
	parser.add_argument('-w', '--weighted_map', help = 'Type: bool. If set to true then Y vector are transformed into weighted map instead of one hot vectors',
		action = 'store_true')

	return parser


def choose_metric(function_metric):

    if function_metric == 'Dice':
        return Dice
    elif function_metric == 'Sorensen_Dice':
        return Sorensen_Dice
    elif function_metric == 'matthews_correlation':
    	return matthews_correlation	
    else: 
        return F1_score


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
	evaluate_metric = arg.evaluate_metric
	weighted_map = arg.weighted_map

	try:
		name_desktop = os.environ['COMPUTERNAME']
		if name_desktop == 'ICH03626':
			bug = True 
	except KeyError:
		bug = False

	# Now we load data for generator via the sampling info list 
	cut_name = name_cut(cut_axis) 
	sampling_info_path = choosing_savename(bug,dimension,'Sampling')
	sampling_info_path = sampling_info_path + '_op_' + str(operator) + cut_name + '.npz'
	
	if not bug:
		sampling_data = np.load(sampling_info_path)['a']
		patient_for_pred = sampling_data[2][0]
		full_path = sampling_data[2][1]

		print('Patients for predictions : {0} / path to prediction : {1} '.format(patient_for_pred,full_path))
		# datagen_test = DataGenerator(sampling_data, 'test', cut_axis, operator, all_image)

		with tf.Session(config=tf.ConfigProto(device_count={ "CPU": 24 },inter_op_parallelism_threads=24,intra_op_parallelism_threads=24)) as sess:
			
			# Setting session, plot data and NN model 
			K.tensorflow_backend.set_session(sess)
			# Loading model
			model_path = '../models/{}_{}.h5'.format(model_str,function_metric)
			if function_metric == 'acc':
				model = load_model(model_path)
			else: 
				model = load_model(model_path,custom_objects={function_metric : choose_metric(function_metric), 'loss':weighted_categorical_crossentropy()})
			
			print('Choosen model : {0} with metric {1} '.format(model_str,function_metric))

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

			# Setting up evaluation variables 
			Eval_list = []
			summary_list = []

			for i in range(8):

				# Selecting patient 
				print('\nCurrent patient : ', i+1)
				current_MRI,current_Seg = list_X[i],list_Y[i]

				# Running preds for this patient 
				current_pred = model.predict(current_MRI,verbose=1)

				# Reshaping data
				current_MRI = np.reshape(current_MRI,(-1,416,416))

				if not weighted_map: 
					current_pred = current_pred[:,:,:,1] 
					current_Seg = current_Seg[:,:,:,1]
				else: 
					current_pred = current_pred[:,:,:,0] 
					current_Seg = current_Seg[:,:,:,0]

				print(current_Seg.shape,current_pred.shape,current_MRI.shape)
				
				# Calculating score for each frame: 
				length = current_MRI.shape[0]
				eval_patient = []

				for j in tqdm(range(length)):

					current_pred_frame = current_pred[j,:,:]
					
					if weighted_map: 

						binary_pred = np.zeros_like(current_pred_frame)
						p = np.quantile(current_pred_frame,0.999)
						binary_pred[current_pred_frame >= p] = 1 
						current_pred_frame = binary_pred

					current_seg_frame = current_Seg[j,:,:] 

					# Evaluate frames
					current_seg_frame = tf.cast(current_seg_frame, tf.float32)
					current_pred_frame = tf.cast(current_pred_frame, tf.float32) 
					if evaluate_metric == 'Dice':
						score = Sorensen_Dice(current_seg_frame,current_pred_frame)
					else: 
						score = matthews_correlation(current_seg_frame,current_pred_frame)
					eval_patient.append(score.eval())

				Eval_list.append(eval_patient)
				np.savez_compressed('../figure/data_pred_{0}_{1}'.format(model_str,function_metric),a=Eval_list)
		
	else: 

		Eval_list = np.load('C:/Users/mbachelo/Desktop/Curie/Code/Cassandre/Figures/data_pred_{0}_{1}.npz'.format(model_str,function_metric))['a']
		fig = plt.figure(figsize=(10,15))
		
		for i in range(8):

			current_patient_score = Eval_list[i]
			print(len(current_patient_score))

			subplot_index = int('24{}'.format(i+1))
			ax = fig.add_subplot(subplot_index)
			
			bins,mu,sigma = plot_histogram(current_patient_score)
			ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
			
			ax.title.set_text('Patient n° {}'.format(i+1))
			ax.set_xlim([0,1])

		fig.suptitle('Patient prediction histograms', fontsize=16)
		plt.show()

