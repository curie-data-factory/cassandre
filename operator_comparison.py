########################################################## READ ME ##########################################################
#                                                                                                                           #
#                                       Compare segmentations between two operators                                         #
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
from tqdm import tqdm
from dltk.io.preprocessing import whitening,normalise_zero_one
from dltk.io.augmentation import flip
from sklearn.model_selection import train_test_split

import numpy as np
import warnings,os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf 
import random as rd
import math as m
import tensorflow as tf

from image_processing import *
from utils import *


########################## index comparison functions ##########################


def comparing_relevance_list(final_relevance_list):
	""" Return frame index of matching tumor segmentation for both operator & frame that are exclusive to one operator 

	Args:
	final_relevance_list (list of lists of lists): For both operator & for each patient, contain the list of frame's index that have tumors   

	Returns:
	list of list, list of list: matching frame index for each patients & exclusive frame index for each patient   
	"""

	# We get both relevance lists of differents operators & variable initialisation 
	relevance_list_op_1,relevance_list_op_2 = final_relevance_list[0],final_relevance_list[1]
	number_of_patients = len(relevance_list_op_1) 
	matching_index_list = []
	exclusive_index_list = []

	# finding matching and excluding index 
	for i in range(number_of_patients):

		current_patient_tumors = [relevance_list_op_1[i],relevance_list_op_2[i]]
		exclusive_indexes = get_sublist(current_patient_tumors[0],current_patient_tumors[1])
		
		# if exclusive_index is not an empty list, we calculate its size compared to the matching_index_list
		if exclusive_indexes != []: 

			lenght_1,lenght_2 = len(current_patient_tumors[0]),len(current_patient_tumors[1])

			# If the first list has a superior length
			if lenght_1 > lenght_2: 

				# We save the patient number, the operator that tagged additional frame (here its op 1) the ratio & indexes
				ratio = lenght_2 / lenght_1
				exclusive_index_list.append([i,1,ratio,exclusive_indexes])	
				matching_index_list.append(current_patient_tumors[1])

			# If the second list has a superior length
			else: 

				# We save the patient number, the operator that tagged additional frame (here its op 1) the ratio & indexes
				ratio = lenght_1 / lenght_2
				exclusive_index_list.append([i,2,ratio,exclusive_indexes])	
				matching_index_list.append(current_patient_tumors[0])
		
		else: 

			matching_index_list.append(current_patient_tumors[0])

	print(matching_index_list,'\n')

	return matching_index_list,exclusive_index_list





if __name__ == '__main__':
	

	try:
		name_desktop = os.environ['COMPUTERNAME']
		if name_desktop == 'ICH03626':
			bug = True 
	except KeyError:
		bug = False


	# Get saved data or create the raw dataset entirely
	final_relevance_list,Y_list = pre_processing(False,1,2,bug,0,False,True)
	matching_index_list,exclusive_index_list = comparing_relevance_list(final_relevance_list)

	number_of_patients = len(matching_index_list)
	eval_patient = []

	with tf.Session(config=tf.ConfigProto(device_count={ "CPU": 24 },inter_op_parallelism_threads=24,intra_op_parallelism_threads=24)) as sess:

		# Setting session, plot data and NN model 
		K.tensorflow_backend.set_session(sess)

		for i in range(number_of_patients):

			current_indexes = matching_index_list[i]
			current_volume_op_1 = Y_list[i][1]
			current_volume_op_2 = Y_list[i][2]
			volume = len(current_indexes)
			eval_patient.append([])
			
			for j in range(volume):

				frame_op_1 = get_frame(2,current_volume_op_1,current_indexes[j])
				frame_op_2 = get_frame(2,current_volume_op_2,current_indexes[j])
				frame_op_1 = resize_image(frame_op_1,(416,416))
				frame_op_2 = resize_image(frame_op_2,(416,416))
				frame_op_1 = np.clip(frame_op_1,0,1)
				frame_op_2 = np.clip(frame_op_2,0,1)		

				unique,counts = np.unique(frame_op_1,return_counts=True)
				unique2,counts2 = np.unique(frame_op_2,return_counts=True)
				print(dict(zip(unique,counts)))
				print(dict(zip(unique2,counts2)))
				print('=======================')

				# plt.subplot(1,2,1)
				# plt.imshow(frame_op_1.T,origin='lower',cmap='gray')
				# plt.subplot(1,2,2)
				# plt.imshow(frame_op_2.T,origin='lower',cmap='gray')
				# plt.show()
				# plt.close()

				frame_op_1 = tf.cast(frame_op_1, tf.float32)
				frame_op_2 = tf.cast(frame_op_2, tf.float32)
				score = Sorensen_Dice(frame_op_1,frame_op_2)
				eval_patient[i].append(score.eval())

			mean_dice = np.mean(eval_patient[i],axis=0)

			# We take into account the fact that there are potentially exclusive frames for that patient for a given operator 
			if len(exclusive_index_list) > 0:

				exclusive_frame = exclusive_index_list[0][0]
				
				if exclusive_frame == i:
					# we multipliate the mean dice by the proportion of matching frame 
					print('Found exclusive frames from one operator for patient {}'.format(i+1))
					mean_dice *= exclusive_index_list[0][2]
					exclusive_index_list.pop(0)
			
			print('Dice coefficient for patient number {0}: {1}'.format(i+1,mean_dice))



