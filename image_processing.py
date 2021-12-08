########################################################## READ ME ##########################################################
#                                                                                                                           #
#                                                    Image Preprocessing                                                    #
#                 This file has all the core preprocessing functions for data extraction & dimension management             #
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
from skimage.io import imshow

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

from segmentation_models import *
from utils import *


########################## Raw data extraction functions ##########################


def data_extraction():
	"""Return two list containing the MRI data (ndarray)

	Args:
	label (list): list of folder label for data extraction 

	Returns:
	(list,list) list cointaining non segmented image data and another one containing all three segmented images data (Y op1 / Y 40 and Y 40 op2)
	"""

	# Get path regarding the machine we are working on (desktop or kubernetes servers)
	try:
		name_desktop = os.environ['COMPUTERNAME']
		if name_desktop == 'ICH03626':
			path = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/' 
	except KeyError:
		path = '/var/www/data/'
	

	# Initialize variables 
	X_list = []
	Y_list = []
	labels = ['RawVolume','RoiVolume']

	for dirs, folders, files in os.walk(path): 

		if 'cassandre' in dirs or 'datasets' in dirs or 'lost' in dirs or 'figure' in dirs or 'models' in dirs:
			pass

		# Extract the X array 
		elif labels[0] in dirs:
			print(dirs)
			X = load_nii(dirs,files[0])
			print('Done for file ',files[0])
			X_list.append(X)

		# Extract the Y arrays
		elif labels[1] in dirs:
			print(dirs)
			Y_tempo = []
			for i in range(len(files)): 

				# Mandatory block : otherwise files can sometimes not be opened for unknown reason 
				# (get error: nibabel.filebasedimages.ImageFileError: Cannot work out file type of...)
				try:
					Y = load_nii(dirs,files[i])
				except Exception :
					pass

				Y_tempo.append(Y)
				print('Done for file ',files[i])
			Y_list.append(Y_tempo)

		else: 
			pass

	return X_list,Y_list 


def get_preprocessed_data(dimension,bug,operator,cut_axis):
	"""Return two ndarrays to be converted in Xe path where the dataset will be loaded

	Returns:_train / Y_train / X_test / Y_test 
	It loads them from a .npz file.

	Args:
	bug (bool): for debugging. It changes the saving path
	np.ndarray,np.ndarray : Raw dataset for segmentation
	"""

	# Getting dataset name given arguments
	string = 'dataset' 
	str_cut = name_cut(cut_axis)

	if dimension: 
		name_X = string + '_X_3D.npz'
		name_Y = string + '_Y_3D.npz'
	else: 
		name_X = string + '_X_2D_op_' + str(operator) + str_cut + '.npz'
		name_Y = string + '_Y_2D' + str_cut + '.npz'

	if bug: 
		savingpath = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/' 
	else: 
		savingpath = '../datasets/' 

	# Loading data 
	X_path = os.path.join(savingpath,name_X)
	X_data = np.load(X_path)
	X = X_data['a']

	# Because there are several elements in Y its more complicated 
	Y_path = os.path.join(savingpath,name_Y)
	Ys = np.load(Y_path)
	if operator == 0:
		Y = Ys['a']
	elif operator == 1:
		Y = Ys['b']
	else: 
		Y = Ys['c']

	return X,Y


########################## Functions for train val & test processing ##########################


def split_train_test(images,segmented_img,validation_split,dimension): 
	"""Create X_train / y_train / X_test / y_test dataset Used for classical mini batch training 

	Args:
	images (list of ndarray): list for X_test & X_train
	segmented_img (list of ndarray): list for Y_test & Y_train
	validation_split (float): Proportion of data that will be used for validation 

	Returns:
	np.ndarray * 4 : datasets for segmentation
	"""

	# We initialize variable We take some patients for train and other for tests
	number_of_patients = len(images)
	number_of_patients_for_test = m.floor(number_of_patients*validation_split)
	number_of_patients_for_train = number_of_patients - number_of_patients_for_test
	print('Number of patients for train : {}'.format(number_of_patients_for_train))
	print('Number of patients for test : {}'.format(number_of_patients_for_test))

	# Define random samples amongst all patients in order to have two distinct groups of patients for train and test.
	# That way the frames / volumes will not be mixed for training and validation
	patients_list = list(range(0,number_of_patients))
	patients_for_train = rd.sample(patients_list,number_of_patients_for_train)
	patients_for_test = get_sublist(patients_list, patients_for_train)
	init_index_train = patients_for_train.pop()
	init_index_test = patients_for_test.pop()
	X_train = images[init_index_train]
	y_train = segmented_img[init_index_train]
	X_test = images[init_index_test]
	y_test = segmented_img[init_index_test]

	# Now we create the train and test set
	for patients in patients_for_train:
		X_train = np.concatenate((X_train,images[patients]),axis=0)
		y_train = np.concatenate((y_train,segmented_img[patients]),axis=0)
	for patients in patients_for_test:
		X_test = np.concatenate((X_test,images[patients]),axis=0)
		y_test = np.concatenate((y_test,segmented_img[patients]),axis=0)

	# We expand the dims along the last axis
	X_train = np.expanding_dimension(X_train,dimension)
	# y_train = np.expanding_dimension(y_train,dimension)
	X_test = np.expanding_dimension(X_test,dimension)
	# y_test = np.expanding_dimension(y_test,dimension)	

	print('New shape of images dataset : train {0} & test {1}'.format(X_train.shape,X_test.shape))

	return X_train,y_train,X_test,y_test


def split_train_test_in_dir(images,segmented_img,validation_split,test_size,cut_axis,operator,dimension,bug): 
	"""Create X_train / y_train / X_test / y_test dataset Used for custom generator usage 

	Args:
	images (list of ndarray): list for X_test & X_train
	segmented_img (list of ndarray): list for Y_test & Y_train
	validation_split (float): Proportion of data that will be used for validation 

	Returns:
	np.ndarray * 4 : datasets for segmentation
	"""

	# We first perform data augmentation: 
	print('Performing data augmentation ...')
	final_images, final_segmented_img = data_augmentation(images,segmented_img)
	print('Done\n')

	# We initialize variable We take some patients for train and other for tests
	number_of_patients = len(final_images)
	number_of_patients_for_test = m.floor(number_of_patients * test_size)
	number_of_patients_for_train = number_of_patients - number_of_patients_for_test
	print('Number of patients for train (including validation split): {}'.format(number_of_patients_for_train))
	print('Number of patients for test : {}'.format(number_of_patients_for_test))
	

	# Define random samples amongst all patients in order to have two distinct groups of patients for train and test.
	# That way the frames / volumes will not be mixed for training / testing and validation
	patients_list = list(range(0,number_of_patients))
	patients_for_train = rd.sample(patients_list,number_of_patients_for_train)

	# We define the validation split only when we have define the whole training set
	number_of_patients_for_val = m.floor(number_of_patients_for_train * validation_split)
	number_of_patients_for_train -= number_of_patients_for_val
	print('Number of patients for val : {}'.format(number_of_patients_for_val))
	
	# We define the patients that will be used for test and validation  
	patients_for_test = get_sublist(patients_list, patients_for_train)
	patients_for_val = rd.sample(patients_for_train,number_of_patients_for_val)
	patients_for_train = get_sublist(patients_for_train,patients_for_val)

	# Creating directories for train test and val values
	savingpath_train = choosing_savename(bug,dimension,'train')
	savingpath_test = choosing_savename(bug,dimension,'test')
	savingpath_val = choosing_savename(bug,dimension,'val')

	# We save data for each patient in it's own directory 
	saving_list = [[patients_for_train,savingpath_train],[patients_for_val,savingpath_val],[patients_for_test,savingpath_test]]

	for i in range(3):
		current_list,current_path = saving_list[i][0],saving_list[i][1]
		print(current_path)
		X_dim = saving_patients_in_dir(current_list,final_images,final_segmented_img,current_path,cut_axis,operator,dimension)

	# # Now we change the values of the number_of_patients_for_ variables because there has been data augmentation
	# number_of_patients_for_train *= 2
	# number_of_patients_for_test *= 2
	# number_of_patients_for_val *= 2

	# print('Data augmentation successfully performed. Total number of images :')
	# print('Train : {}'.format(number_of_patients_for_train))
	# print('Test : {}'.format(number_of_patients_for_test))
	# print('Val : {}'.format(number_of_patients_for_val))

	# Saving sampling informations (length of lists in the directory and saving path)
	sampling_info = np.array([[number_of_patients_for_train,savingpath_train],[number_of_patients_for_val,savingpath_val],[number_of_patients_for_test,savingpath_test],[X_dim]])
	sampling_info_path = choosing_savename(bug,dimension,'Sampling')
	cut_name = name_cut(cut_axis) 
	sampling_info_path = sampling_info_path + '_op_' + str(operator) + cut_name
	np.savez_compressed(sampling_info_path, a=sampling_info)


########################## Main ##########################


def pre_processing(dimension,operator,cut_axis,bug,padding,weighted_map,comparing_op=False): 
	"""Return two ndarrays to be converted in X_train / Y_train / X_test / Y_test 
	It will save those array in a .npz file. Use this function if you have added data and have to recreate new datasets

	Args:
	dimension (bool): choose the segmentation type (2D or 3D). It will change the way the array is returned 
	operator (int): choose the operator for the creation of the Y array (label) 
	cut_axis (int): choose the axis to be cut for 2D segmentation (only)
	bug (bool): for debugging. It changes the path where the dataset will be saved 
	padding (int): Choose the number of frames you want to add to the dataset with relevant ones 
    weighted_map (bool): If set to True then transfom Y array into weighted maps of shape (max_dim,max_dim,1) instead of one hot vector of shape (max_dim,max_dim,2)  

	Returns:
	np.ndarray,np.ndarray : Raw datasets for segmentation
	"""

	# retrieving raw data from folder & variable definitions
	X_list,Y_list = data_extraction()
	number_of_patients = len(X_list)
	Y_list_finale = []
	deal_only_with_Y = False
	base_operator = operator
	op = str(base_operator)
	str_cut = name_cut(cut_axis)

	# We create the dimension of the future arrays 
	if dimension:
		max_width,max_height = check_shape(X_list,1)
	else: 
		max_width,max_height = check_shape(X_list,cut_axis)
	max_dim = max(max_width,max_height)
	
	# We want to have a cube (or a square) with an even size of voxel, in order to have a clean segmentation 
	if max_dim%2 != 0:
		max_dim += 1 
	else: 
		pass

	if dimension:
		print('\nFrame number : ',X_list[0].shape[1] ,' Future images dimensions : ', (max_dim,max_dim,max_dim))
	else: 
		print('\nFuture images dimensions : ', (max_dim,max_dim),'\n')

	# Building relevance lists for both operators
	already_warned = False
	print('Axis name : ', name_cut(cut_axis))
	print('\nBuilding relevance lists ...')
	relevance_list_op1 = check_relevance(Y_list,cut_axis,1,number_of_patients,padding,already_warned)
	relevance_list_op2 = check_relevance(Y_list,cut_axis,2,number_of_patients,padding,already_warned)
	final_relevance_list = [relevance_list_op1,relevance_list_op2]

	# For operator_comparison.py script only 
	if comparing_op: 
		return final_relevance_list,Y_list

	# Now we create the whole 3 Y datasets (preprocessed) and the X dataset too, we return the X and one Y choosen by the argument operator
	for j in range(3):

		# This loop is made for going through all 3 Y images, so we set operator to j. 

		if j == 0: 

			# 3D segmentation
			if dimension:

				# Images processing 
				images,segmented_img = resizing_and_normalize_3D(X_list,Y_list,number_of_patients,max_dim,j,deal_only_with_Y)
				# images = np.reshape(images,(-1,max_dim,max_dim,max_dim))
				# segmented_img = np.reshape(segmented_img,(-1,max_dim,max_dim,max_dim))
				# We extracted and preprocessed the X value, now we save it: 
				if bug: 
					savingpath = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/dataset_X_3D'
				else: 
					savingpath = '../datasets/dataset_X_3D'
			
			# 2D segmentation
			else:

				# Images processing
				images,segmented_img = resizing_and_normalize_2D(X_list,Y_list,number_of_patients,max_dim,j,cut_axis,deal_only_with_Y,
					base_operator,padding,final_relevance_list,weighted_map)
				# images = np.reshape(images,(-1,max_dim,max_dim))
				# segmented_img = np.reshape(segmented_img,(-1,max_dim,max_dim))
				# We extracted and preprocessed the X value, now we save it: 
				if bug: 
					savingpath = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/dataset_X_2D_op_' + op + str_cut
				else: 
					savingpath = '../datasets/dataset_X_2D_op_' + op + str_cut
			
			np.savez_compressed(savingpath, a=images)
			print('\n==========================================================================================================================================================')
			print('\nX Data saved successfully\n')
				
			# we append the final Y dataset to the final list (just to save it later) and we reset deal_only_with_Y variable (because X was saved)
			Y_list_finale.append(segmented_img)
			deal_only_with_Y = True

		# At this point X is saved so we just need to create the other two Y datasets
		else:

			# 3D segmentation
			if dimension:  
				segmented_img = resizing_and_normalize_3D(X_list,Y_list,number_of_patients,max_dim,j,deal_only_with_Y)
				# segmented_img = np.reshape(segmented_img,(-1,max_dim,max_dim,max_dim))
				Y_list_finale.append(segmented_img)
			else: 
				segmented_img = resizing_and_normalize_2D(X_list,Y_list,number_of_patients,max_dim,j,cut_axis,deal_only_with_Y,
					base_operator,padding,final_relevance_list,weighted_map)
				# segmented_img = np.reshape(segmented_img,(-1,max_dim,max_dim))
				Y_list_finale.append(segmented_img)

	# saving path options block
	if dimension:
		if bug: 
			savingpath = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/dataset_Y_3D'
		else: 
			savingpath = '../datasets/dataset_Y_3D'
	else: 
		if bug: 
			savingpath = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/dataset_Y_2D'+ str_cut + '.npz'
		else: 
			savingpath = '../datasets/dataset_Y_2D' + str_cut + '.npz'

	# Saving Ys 
	Y,Y_40,Y_40_op2 = Y_list_finale[0],Y_list_finale[1],Y_list_finale[2]
	np.savez_compressed(savingpath, a=Y, b=Y_40, c=Y_40_op2)
	print('\n==========================================================================================================================================================')
	print('\nY Data saved successfully\n')

	# Choosing Y for tests: 
	if operator == 0: 
		segmented_img = Y
	elif operator == 1: 
		segmented_img = Y_40
	else: 
		segmented_img = Y_40_op2
	
	return images,segmented_img



########################## Debug functions ##########################


def test_coherence(): 
	"""Check data integrity 
	"""

	savingpath_train = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/train_2D'
	savingpath_test = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/test_2D'
	savingpath_val = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/val_2D'

	path_list1 = [savingpath_train,savingpath_val,savingpath_test]
	path_list2 = ['MRI','Segmentation']

	for i in range(3): 
		savingpath_X = os.path.join(path_list1[i],path_list2[0],'patient_1_op_1_axial.npz')
		savingpath_Y = os.path.join(path_list1[i],path_list2[1],'patient_1_op_1_axial.npz')
		X = np.load(savingpath_X)['a']
		Y = np.load(savingpath_Y)['a']
		
		X = X_train_transform(X)

		print(X.shape,Y.shape)
		maxi,mini,mean,std = np.max(X),np.min(X),np.mean(X),np.std(X)
		unique,counts = np.unique(Y, return_counts=True)
	

	sampling_info = np.load('C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/Sampling_2D_op_1_axial.npz')['a']

	print(sampling_info)

	y = np.reshape(Y,(-1,416,416,2))
	y2 = np.reshape(X,(-1,416,416))	

	length = min(y.shape[0],y2.shape[0])
	sample_size = 4
	list_of_index = rd.sample(range(0,length),sample_size)
	plt.figure(figsize=(10,10))

	for i,sample in enumerate(list_of_index):
		
		slice = get_frame(0,y,sample)
		# slice_class_1 = slice.T[0]
		slice_class_2 = slice.T[1]
		slice_test = get_frame(0,y2,sample)
		
		# # Check voxel distribution
		# mu, sigma = np.mean(slice_test), np.std(slice_test)
		# count, bins, ignored = plt.hist(slice_test, 30, density=True) 
		# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
		# plt.suptitle("Voxel value distribution for a random slice")
		# plt.show()
		# plt.close()

		unique,counts = np.unique(slice_class_2, return_counts=True)
		# print(dict(zip(unique,counts)))
		print("Relevant slice ",sample)

		plt.subplot(4,2,i*2+1)
		plt.imshow(slice_test.T, cmap="gray", origin="lower")
		
		# plt.subplot(4,3,i*3+2)
		# plt.imshow(slice_class_1, cmap="gray", origin="lower")

		plt.subplot(4,2,i*2+2)
		plt.imshow(slice_class_2, cmap="gray", origin="lower")

	plt.suptitle("Y vector vizualisation")
	plt.show()



def check_Y(bug,dimension):

	path_train = choosing_savename(bug,dimension,'train')
	path_test = choosing_savename(bug,dimension,'test')
	path_val = choosing_savename(bug,dimension,'val')

	path_train = os.path.join(path_train,'Segmentation')
	path_test = os.path.join(path_test,'Segmentation')
	path_val = os.path.join(path_val,'Segmentation')

	Path_list = [[path_train,2],[path_test,2],[path_val,2]]
	invalid_data,valid_data,total_frame = 0,0,0
	compteur_plot = 1 
	plt.figure(figsize=(15, 7))

	for k in range(len(Path_list)):

		path,number_of_patients = Path_list[k][0],Path_list[k][1]

		for i in range(number_of_patients): 

			path_to_patient = 'patient_' + str(i+1) + '_op_1_axial.npz'
			path_to_patient = os.path.join(path,path_to_patient)
			Y = np.load(path_to_patient)['a']

			length = Y.shape[0]
			total_frame += length
			sample = rd.randint(1,length-1)
			current_Y = get_frame(0,Y,sample)
			class_0 = get_frame(2,current_Y,0)
			class_1 = get_frame(2,current_Y,1)

			plt.subplot(4,3,compteur_plot)
			imshow(class_0.T)
			plt.subplot(4,3,compteur_plot+1)
			imshow(class_1.T)
			compteur_plot += 2

	plt.show()


def check_all_X():

	path_train = '../datasets/train_2D/MRI'
	path_test = '../datasets/test_2D/MRI'
	path_val = '../datasets/val_2D/MRI'

	Path_list = [[path_train,36],[path_test,4],[path_val,4]]

	plt.figure(figsize=(15, 7))
	compteur_plot = 1 

	for k in range(len(Path_list)):

		path,number_of_patients = Path_list[k][0],Path_list[k][1]

		for i in range(number_of_patients): 

			path_to_patient = 'patient_' + str(i+1) + '_op_1_axial.npz'
			path_to_patient = os.path.join(path,path_to_patient)
			X = np.load(path_to_patient)['a']
			length = X.shape[0]
			X = np.reshape(X,(-1,416,416))
			
			sample = rd.randint(1,length-1)
			slice_X = get_frame(0,X,sample)
			slice_X = X_train_transform(slice_X)

			plt.subplot(4,11,compteur_plot)
			plt.imshow(slice_X.T, cmap="gray", origin="lower")
			compteur_plot += 1 

	plt.savefig('../figure/test_X.png')

if __name__ == '__main__':

	test_coherence()