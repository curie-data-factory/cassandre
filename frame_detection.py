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

from image_processing import *
from utils import *


########################## Raw data extraction functions ##########################


def frame_pre_processing(dimension,operator,cut_axis): 
	"""Return two ndarrays to be converted in X_train / Y_train / X_test / Y_test 
	It will save those array in a .npz file. Use this function if you have added data and have to recreate new datasets

	Args:
	dimension (bool): choose the segmentation type (2D or 3D). It will change the way the array is returned 
	operator (int): choose the operator for the creation of the Y array (label) 
	
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

	# Building relevance lists...
	already_warned = False
	print('Axis name : ', str_cut)
	print('\nBuilding relevance lists ...')
	
	if operator < 2:
		healthy_list,relevance_list = check_relevance(Y_list,cut_axis,1,number_of_patients,0,already_warned)
	else: 
		healthy_list,relevance_list = check_relevance(Y_list,cut_axis,2,number_of_patients,0,already_warned)

	


