########################################################## READ ME ##########################################################
#                                                                                                                           #
#                 				          Utility functions used in Cassandre's project                                     #
#                                                                                                                           #
#############################################################################################################################


########################## Imports ##########################
 

import numpy as np
import nibabel as nib
import os,argparse,keras,warnings
import SimpleITK as sitk
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math as m

from keras import backend as K
from keras.utils import to_categorical

from dltk.io.preprocessing import whitening,normalise_zero_one

from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt


########################## Basic functions ##########################


def load_nii(path,file):
    """Load an .nii.gz file via nibabel 

    Args:
    path (str): path to folder containing file 
    file (str): nii.gz file 

    Returns:
    np.ndarray: image data 
    """
    
    final_path = os.path.join(path,file)
    epi_img = nib.load(final_path)
    epi_img_data = epi_img.get_data()
    
    return epi_img_data


def load_sitk(path,file):
    """Load an .nii.gz file via sitk  

    Args:
    path (str): path to folder containing file 
    file (str): nii.gz file 

    Returns:
    np.ndarray: image data 
    """

    final_path = os.path.join(path,file)
    epi_img = sitk.ReadImage(final_path)
    epi_img_data = sitk.GetArrayFromImage(epi_img)
    
    return epi_img_data

def resize_image(image, img_size=(64, 64, 64)):
    """Image resizing. Resizes image by cropping or padding dimension
    to fit specified size.

    Args:
    image (np.ndarray): image to be resized
    img_size (list or tuple): new image size

    Returns:
    np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
    'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, 'minimum' )


def name_cut(cut_axis):
    """Get the scientific name of current cut (for saving purposes).
    """

    if cut_axis == 0: 
        str_axis = '_sagittal'
    elif cut_axis == 1: 
        str_axis = '_coronal'
    else: 
        str_axis = '_axial'

    return str_axis


def get_sublist(list1, list2): 
    """Given two lists [a,b,c,d] and [a,c], return the list [b,d].
    """

    list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2] 
    
    return list_dif 


def check_shape(input_list,cut_axis):
    """return the highest dimension for both axis in a list of ndarray

    Args:
    input_list (list): list of nifti images (2D images i.e. np.ndarray)
    cut_axis (int): axis on which the image will be sliced

    Returns:
    (int,int) highest dim for axis 0 and 1 
    """

    n = len(input_list)
    list_axis_0, list_axis_1 = [],[]

    for i in range(n):
        
        if cut_axis == 0:
            list_axis_0.append(input_list[i].shape[1])
            list_axis_1.append(input_list[i].shape[2])
        if cut_axis == 1:
            list_axis_0.append(input_list[i].shape[0])
            list_axis_1.append(input_list[i].shape[2])
        else:
            list_axis_0.append(input_list[i].shape[0])
            list_axis_1.append(input_list[i].shape[1])

    return max(list_axis_0), max(list_axis_1)


def create_saving_folder(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)


def expanding_dimension(X,dimension):

    if dimension: 
        return np.expand_dims(X,axis=4)
    else: 
        return np.expand_dims(X,axis=3)


def choosing_savename(bug,dimension,file_name): 

    # Dimension influence on saving path 
    if dimension: 
        extention_name = '_3D'
    else: 
        extention_name = '_2D'

    # Bug influence on saving path 
    if bug: 
        core_path = 'C:/Users/mbachelo/Desktop/Curie/Code/Bagdad/data/datasets/'
    else: 
        core_path = '../datasets/'

    return core_path + file_name + extention_name


def check_neighbours(value,next_value): 
    """Check if those two values are neighbours
    """

    target_value = next_value - 1
    if value ==target_value: 
        return True 
    else: 
        return False


def choose_classes(weighted_map):

    classes = 0
    
    if weighted_map:
        classes += 1 
    else: 
        classes += 2

    return classes


########################## Frame processing functions ##########################


def get_frame(cut_axis,img_data,index):
    """ return a frame from an image along the given axis and index 

    Args:
    cut_axis (int): Axis on which the slices will be retrieved
    img_data (np.ndarray): 3D array representing the images 
    index (int): index to get frame 

    Returns:
    ndarray. Frame from img_data
    """

    if cut_axis == 0:
        return img_data[index, :, :]
    elif cut_axis == 1:
        return img_data[:, index, :]
    else: 
        return img_data[:, :, index]


def sum_positive_values(img): 
    """ Sums the number of occurence of positive voxels inside a given img 

    Returns:
    int: the values of positives voxels

    """

    unique,counts = np.unique(img, return_counts=True)
    value_dict = dict(zip(unique,counts))
    total_sum = 0 

    # Now we look at the number of positive voxels (1 and greater) and we sum it
    for key in value_dict: 
        if key == 0: 
            pass 
        else: 
            tempo_sum = value_dict[key]
            total_sum += tempo_sum

    return total_sum


def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):

    """Generate weight maps as specified in the U-Net paper for boolean mask.

    Args:
    mask (ndarray):  2D array of shape (height, width) representing binary mask of objects.
    wc (dict): weight classes.
    w0 (int):  Border weight parameter.
    sigma (int): Border width parameter.

    Returns: 
    ndarray: Training weights. A 2D array of shape (height, width).
    """
    
    # We create a labeled array (all connected regions are assigned the same int value)
    labels = label(y)
    
    # We create two variables that separates backgrounds (no_labels) from tumors (labels_ids, which is a list like this [1,2,3,...,n])
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]
    
    if len(label_ids) > 1:
        
        # We create one feature map for each non zero label (so here we have 5 fm)
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))
        
        # We create the distance between labeled pixel & others
        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)
        
        # We select the first class (background) and the second (tumor)
        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        
        # We apply the paper formula on those two class
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        
        # We add weigths for each pixels 
        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
    else:
        # If labels_ids has a length of 1 (ie labels_ids=[1]) then we cant perform transformation above, so we return the input 
        w = y
    
    return w


def get_padding_index(relevance_list,padding): 
    """Get a list of index next to relevant cut in order to pad our data with cut on which there is no cancer

    Args:
    relevance_list (list): list of index which indicate cut with tumors on it 
    padding (int): number of padding cut that we'll incorporate in the dataset

    Returns:
    relevance_list: list of frames'index that need to be kept in the dataset
    """

    # Tumors are volume, so the relevance_list is composed of several series of following integers like this [n,n+1,...,n+12,m,m+1,...,m+22]
    # This list is meant to save those values in a tuple [(n,n+12),(m,m+22)] 
    volume_list = []
    edge_list = []
    length = len(relevance_list)

    # Here volume_start will be equal to n or m  & volume_end will be equal to n+12 or m+22
    volume_start,volume_end = relevance_list[0],0 
    volume_start_index = 0

    for i in range(length-1):

        if check_neighbours(relevance_list[i],relevance_list[i+1]):
            pass
        else:
            
            # If the consecutive values are not neighbours that's mean we got to the end of a given tumor volume 
            volume_end = relevance_list[i]
            edge_tuple = (volume_start,volume_end) 
            edge_list.append(edge_tuple)
            current_volume = relevance_list[volume_start_index:i]
            volume_list.append(current_volume)

            # We update the volume_start & volume_start_index variables 
            volume_start = relevance_list[i+1]
            volume_start_index = i+1

    # Now we deal with the last index: 
    volume_end = relevance_list[length-1]
    edge_tuple = (volume_start,volume_end)
    edge_list.append(edge_tuple)
    current_volume = relevance_list[volume_start_index:length-1]
    volume_list.append(current_volume)

    # Because all volume are stored into the edge_list, we create two padding lists and we put them at the 2 edges of their matching volume 
    length = len(edge_list)
    padding_dict = {}

    for i in range(length):

        # Initialize storage variables
        volume_index = edge_list[i]
        current_volume = volume_list[i]
        index_start = volume_index[0] - padding
        index_end = volume_index[1] + padding + 1

        # Creating padding list 
        padding_sub_list_start = np.arange(index_start,volume_index[0]).tolist()
        padding_sub_list_end = np.arange(volume_index[1],index_end).tolist()

        # Padding current volume 
        padded_volume = padding_sub_list_start + current_volume + padding_sub_list_end
        padding_dict[volume_index] = padded_volume 

    # reconstruct the relevance list 
    final_relevance_list = []    
    for key in padding_dict: 
        final_relevance_list = final_relevance_list + padding_dict[key]

    # Remove duplicate index 
    final_relevance_list = list(dict.fromkeys(final_relevance_list))

    return final_relevance_list


def check_relevance(Y_list,cut_axis,operator,number_of_patients,padding,already_warned,return_healthy_cut=False):
    """ Return a list of index which indicates images with a cancer on it

    Args:
    Y_list (list of list of ndarray): list of list. Sublists at index i has 3 differents ground truth matching X ndarray of X_list at index i 
    cut_axis (int): axis to work with 
    operator (int): operator choosen for ground truth 
    number_of_patients (int): number of patient (there is one relevance list per patients becauses relevant slices may differ from a patient to another)
    already_warned (bool): use for printing messages
    return_healthy_cut (bool): used for frame detection inside a MRI volume

    Returns:
    relevance_list: list of lists. list at index i contains list of relevant slices indexes in MRI i (where there is actually a cancer, ie where there is at least one white pixel)
    """

    if operator == 0: 
        warnings.warn("Cannot work out relevance check with operator 0. Operator changed to 1 (same dataset but 40% threshold applied)")
    else: 
        pass

    relevance_list = []
    relevance_number = 0

    for i in range(number_of_patients): 

        current_Ys = Y_list[i]
        
        if operator == 0:
            Y_tempo = current_Ys[operator+1]
        else: 
            Y_tempo = current_Ys[operator]

        length = Y_tempo.shape[cut_axis]
        current_relevance_list = []

        for j in range(length):

            # If there is a single voxel which value is greater than 0 then it means there is a tumor in this frame
            current_frame = get_frame(cut_axis,Y_tempo,j)
            total_sum = sum_positive_values(current_frame)

            # We only keep frames in which at least 50 voxels are marked as tumor (thus it will be easier for predictions) 
            if total_sum > 50:
                current_relevance_list.append(j)
                relevance_number += 1
            else: 
                pass
                
        # Now we add padding frames to the current relevance list and we append it to relevance_list
        current_relevance_list = get_padding_index(current_relevance_list,padding)
        relevance_list.append(current_relevance_list)

    # Display info on data 
    info_list = [len(relevance_list[i]) for i in range(len(relevance_list))]
    if not already_warned: 
        print('Number of relevant frames for operator {0} (padding excluded) : {1}'.format(operator,relevance_number))
        print('Number of saved frames for operator {0} (padding included) : {1}'.format(operator,sum(info_list)))
        print('Total frame number :',number_of_patients*length)
        if operator == 2: 
            print('\n==========================================================================================================================================================\n')      
    
    if not return_healthy_cut:
        
        return relevance_list
    
    else: 

        healthy_list = []

        for i in range(number_of_patients): 

            current_Ys = Y_list[i]
        
            if operator == 0:
                Y_tempo = current_Ys[operator+1]
            else: 
                Y_tempo = current_Ys[operator]

            length = Y_tempo.shape[cut_axis]
            cancer_cut = relevance_list[i]

            total_frame = range(length)
            all_healthy_cuts = get_sublist(total_frame,cancer_cut)
            number_of_cancerous_frames = len(cancer_cut)
            healthy_cuts = rd.sample(all_healthy_cuts,number_of_cancerous_frames)
            healthy_list.append(healthy_cuts)

        return healthy_list,relevance_list






def creating_dataset(cut_axis,img,current_relevance_list,deal_with_Y=False,weighted_map=False):
    """Slice the image for a 2D segmentation. If it's a label image (a Y array), then perform additionnal processing functions 

    Args:
    cut_axis (int): axis which will be sliced
    img (3D np.array from a nii.gz file): MRI volume or Segmentation volume (to be sliced)  
    current_relevance_list (list): List of slices index 
    deal_with_Y (bool): If set to true then this function will process the extracted 2D slice differently 
    weighted_map (bool): If set to true then transform Y slices into weighted map. Otherwise transform them into one hot vectors

    Returns:
    images: list of 2D images which are guaranted to be all relevant (because there is a cancer in this particuliar slice) 
    """

    images = []

    for index in current_relevance_list:
        
        # We get the frame and if it's a y dataset, we transform it into a one hot vector. 
        slice_j = get_frame(cut_axis,img,index)
        
        # If its a Y slice, we have to preprocess it differently 
        if deal_with_Y: 
            
            # First clip value in order to obtain a binary mask
            slice_j = np.clip(slice_j,0,1)
            
            if not weighted_map: 

                # First processing option: convert it into a one hot vector (with to_categorical) Y has now a shape of (height,width,2) 
                slice_j = to_categorical(slice_j)

            else:

                # Second processing option: convert it into a feature map. The final shape is (height,width,1)
                slice_j = unet_weight_map(slice_j, wc={0:1,1:10})
                slice_j = expanding_dimension(slice_j,False)                  
        
        images.append(slice_j)

    return images


########################## Data Augmentation & Folder management ##########################


def data_augmentation(images,segmented_img): 
    """Perform data augmentation on MRI volume and matching segmentations

    Args:
    images (list of ndarray): list of MRI volumes of dims (None,max_dim,max_dim)
    segmented_img (list of ndarray): list of Segmentation volumes of dims (None,max_dim,max_dim,2)

    Returns:
    (list,list): MRI volume and Segmentation after datga augmentation
    """
    final_images = []
    final_segmented_img = []

    for i in range(len(images)):

        current_MRI = images[i]
        current_segmentation = segmented_img[i]

        flip_X_ax_1 = np.flip(current_MRI,axis=1)
        flip_Y_ax_1 = np.flip(current_segmentation,axis=1)

        print(flip_X_ax_1.shape)
        print(flip_Y_ax_1.shape)

        final_images.append(current_MRI)
        final_images.append(flip_X_ax_1)

        final_segmented_img.append(current_segmentation)
        final_segmented_img.append(flip_Y_ax_1)

    return final_images, final_segmented_img


def saving_patients_in_dir(patient_list,final_images,final_segmented_img,savingpath,cut_axis,operator,dimension):
    """Save the patients in the directory named after savingpath variable

    Args:
    patient_list (list): list of nifti images (2D images i.e. np.ndarray)
    final_images (list of ndarray): MRI data after data augmentation 
    final_segmented_img (list of ndarray) Segmented Label imgs after data augmentation
    savingpath (str): folder path info 
    cut_axis (int): axis on which the image will be sliced
    operator (int): operator for those data (saving purposes)
    dimension (bool): 2D or 3D segmentation 
    Returns:
    ndarray: Dimensions of MRI data for further NN usage
    """

    create_saving_folder(savingpath)

    for i,patients in enumerate(patient_list): 

        # Retrieving data and performing data augmentation 
        (X,Y) = (final_images[patients],final_segmented_img[patients])

        # flip_X_ax_1 = np.flip(X,axis=1)
        # flip_Y_ax_1 = np.flip(Y,axis=1)

        # Expanding dimension for NN if weighted_map
        X = expanding_dimension(X,dimension)
        # flip_X_ax_1 = expanding_dimension(flip_X_ax_1,dimension)

        # current_patient_X = (X,flip_X_ax_1)
        # current_patient_Y = (Y,flip_Y_ax_1)

        # Creating folder and saving names
        cut_name = name_cut(cut_axis) 
        label_folder = os.path.join(savingpath,'Segmentation')
        X_folder = os.path.join(savingpath,'MRI')
        create_saving_folder(label_folder)
        create_saving_folder(X_folder)
        
        # # Saving images and its flipped equivalents
        # for j in range(2): 
        #     # The flipped images count as a different patient 
        #     patient_number = str(i*2 + j + 1)
        #     name = 'patient_' + patient_number + '_op_' + str(operator) + cut_name

        #     path_X = os.path.join(X_folder,name)
        #     path_Y = os.path.join(label_folder,name)

        #     # We save the images
        #     np.savez_compressed(path_X,a=current_patient_X[j])
        #     np.savez_compressed(path_Y,a=current_patient_Y[j]) 

        patient_number = str(i+1)
        name = 'patient_' + patient_number + '_op_' + str(operator) + cut_name
        path_X = os.path.join(X_folder,name)
        path_Y = os.path.join(label_folder,name)

        np.savez_compressed(path_X,a=X)
        np.savez_compressed(path_Y,a=Y) 

        X_dim = X.shape

    return X_dim


########################## Datasets creation functions ##########################


def resizing_and_normalize_2D(X_list,Y_list,number_of_patients,max_dim,operator,cut_axis,deal_only_with_Y,base_operator,padding,final_relevance_list,weighted_map):
    """ Main pre processing function for 2D segmentation. Keep only the relevant slices. It create all three Y dataset and one X dataset only

    Args:
    X_list (list of ndarray): list of raw X values (data from mri converted to ndarray)
    Y_list (list of list of ndarray): list of list. Sublists at index i has 3 differents ground truth matching X ndarray of X_list at index i 
    number_of_patients (int): length of dataset (given by the number of directories)
    max_dim (int): maximum dimension. Even int that will be used to reshape images into cube
    operator (int): operator choosen for ground truth (for Y dataset creation only) 
    deal_only_with_Y (bool): used to save X adn Y datasets 
    cut_axis (int): choose the axis to be cut for 2D segmentation (only)
    base_operator (int): operator choosen for ground truth for X dataset creation
    padding (int): number of frames to add with the relevant one in order to complete the volume 
    final_relevance_list (list of lists of lists): list of the two relevance lists for op 1 and 2.
    weighted_map (bool): If set to True then transfom Y array into weighted maps of shape (max_dim,max_dim,1) instead of one hot vector of shape (max_dim,max_dim,2)  

    Returns:
    ndarray or ndarray,ndarray. X & Y processed data or just Y data (for later save)
    """
    
    # Initialize variables  

    if operator == 0:
        already_warned = False
    else: 
        already_warned = True

    if not already_warned: 
        if base_operator == 0: 
            relevance_list = final_relevance_list[0]
            print('Choosen operator for X and Y : Operator 1 (base dataset: 0)')        
        elif base_operator == 1: 
            relevance_list = final_relevance_list[0]
            print('Choosen operator for X and Y : Operator 1')
        else: 
            relevance_list = final_relevance_list[1]
            print('Choosen operator for X and Y : Operator 2')

        if weighted_map: 
            print('Y data type choosen : weighted map\n')
        else:
            print('Y data type choosen : one hot vector\n')

    # Main loop 
    print('Creating data for operator : {0} ...'.format(operator))
    if not deal_only_with_Y: 
    
        images = []
        segmented_img = []

        for i in range(number_of_patients):
            
            # We create the slices along the given axis for the chosen operator IN THE CASE OF X ONLY 
            current_X,current_Ys = X_list[i],Y_list[i]          
            X_tempo = creating_dataset(cut_axis,current_X,relevance_list[i])
            # The Y relevance_list is independant from the one used for X, it will allow us to create two X datasets with matching relevance slices: one for op 1 one for op 2 
            Y_tempo = creating_dataset(cut_axis,current_Ys[operator],final_relevance_list[0][i],deal_with_Y=True,weighted_map=weighted_map)

            # We pad the array so they all have the same shape. Warning: X and Y may not have the same batch size (depends on operator)
            batch_for_X = len(X_tempo)
            batch_for_Y = len(Y_tempo)

            X_tempo = np.asarray(X_tempo)
            X_tempo = resize_image(X_tempo,(batch_for_X,max_dim,max_dim))

            for k in range(batch_for_Y):
                if not weighted_map: 
                    Y_tempo[k] = resize_image(Y_tempo[k],(max_dim,max_dim,2))
                else: 
                    Y_tempo[k] = resize_image(Y_tempo[k],(max_dim,max_dim,1))
            Y_tempo = np.asarray(Y_tempo)

            # # We append the reshaped image into the list 
            images.append(X_tempo)
            segmented_img.append(Y_tempo)

        return images, segmented_img

    else: 

        segmented_img = []
        for i in range(number_of_patients):

            # We take the images one by one
            current_Ys = Y_list[i]
            if operator == 1 or operator == 0:  
                Y_tempo = creating_dataset(cut_axis,current_Ys[operator],final_relevance_list[0][i],deal_with_Y=True,weighted_map=weighted_map)
            else: 
                Y_tempo = creating_dataset(cut_axis,current_Ys[operator],final_relevance_list[1][i],deal_with_Y=True,weighted_map=weighted_map)           

            # We pad the array so they all have the same shape
            batch_for_Y = len(Y_tempo) 
            for k in range(batch_for_Y):
                if not weighted_map: 
                    Y_tempo[k] = resize_image(Y_tempo[k],(max_dim,max_dim,2))
                else: 
                    Y_tempo[k] = resize_image(Y_tempo[k],(max_dim,max_dim,1))
            Y_tempo = np.asarray(Y_tempo)     

            # We create the dataset
            segmented_img.append(Y_tempo)

        return segmented_img


def resizing_and_normalize_3D(X_list,Y_list,number_of_patients,max_dim,operator,deal_only_with_Y):
    """ Main pre processing function for 3D segmentation 

    Args:
    X_list (list of ndarray): list of raw X values (data from mri converted to ndarray)
    Y_list (list of list of ndarray): list of list. Sublists at index i has 3 differents ground truth matching X ndarray of X_list at index i 
    number_of_patients (int): length of dataset (given by the number of directories)
    max_dim (int): maximum dimension. Even int that will be used to reshape images into cube
    operator (int): operator choosen for ground truth 
    deal_only_with_Y (bool): used to save X adn Y datasets 

    Returns:
    ndarray or ndarray,ndarray. X & Y processed data or just Y data (for later save)
    """

    images, segmented_img = [],[]

    if not deal_only_with_Y: 
        for i in range(number_of_patients):

            # We take the images one by one
            current_X,current_Y = X_list[i],Y_list[i][operator]

            # We pad the array so they all have the same shape
            current_X = resize_image(current_X,(max_dim,max_dim,max_dim))
            current_Y = resize_image(current_Y,(max_dim,max_dim,max_dim))       

            # We create the dataset
            images.append(current_X)
            segmented_img.append(current_Y)

        # We transform into an array the previous list
        images = np.asarray(images)
        segmented_img = np.asarray(segmented_img)
        print('Shape of images dataset : ',images.shape) # (number of patient, number of sample per patient, img width, img height)
        print('Shape of segmented images dataset : ',segmented_img.shape)

        # For a 3D convolution, inputs must be of the following type (tf backend): (batch_size,frames,width,heigth,feature)
        images = np.expand_dims(images,axis=4)
        segmented_img = np.expand_dims(segmented_img,axis=4)
        print('New shape of images dataset : ',images.shape, ' and ', segmented_img.shape)

        return images, segmented_img

    else: 

        for i in range(number_of_patients):

            # We take the images one by one
            current_Y = Y_list[i][operator]

            # We pad the array so they all have the same shape
            current_Y = resize_image(current_Y,(max_dim,max_dim,max_dim))       

            # We create the dataset
            segmented_img.append(current_Y)

        # We transform into an array the previous list
        segmented_img = np.asarray(segmented_img)
        print('Shape of segmented images dataset : ',segmented_img.shape)

        # For a 3D convolution, inputs must be of the following type (tf backend): (batch_size,frames,width,heigth,feature)
        segmented_img = np.expand_dims(segmented_img,axis=4)
        print('New shape of images dataset : ', segmented_img.shape)

        return segmented_img


def X_train_transform(X_train):
    """Return a standardized X_train array (fist z-standardize array, then normalize zero to one all of it)
    Args:
    X_train (ndarray): X_train 

    Returns:
    np.ndarray: Train dataset for segmentation
    """

    X_train = whitening(X_train) 
    X_train = normalise_zero_one(X_train)
            
    return X_train


########################## Metric & loss for NN ##########################


def Sorensen_Dice(y_true,y_pred,smooth=1):
    """Metric for the CNN: it fits well for semantic segmentation. 

    Args:
    y_true (tensorflow -tf- tensor): ground truth given by Radiologists
    y_pred (tf tensor): prediction vector from the CNN
    smooth (int): smoothen the coefficient

    Returns:
    tf tensor: Dice coefficient (float between 0 and 1)
    """
    
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)

    return K.mean((2. * intersection + smooth) / (union + smooth))


def Dice_coeff_loss(y_true, y_pred):
    """Loss for the CNN built via the Sørensen & Dice index

    Args:
    y_true (tf tensor): ground truth given by Radiologists
    y_pred (tf tensor): prediction vector from the CNN

    Returns:
    tf tensor: Dice loss (float between 0 and 1)
    Note: With backpropagation, loss = -Dice_coeff would have worked too. The 1 is just to have a positive loss
    """    

    return 1-Sorensen_Dice(y_true, y_pred)


def Dice(y_true,y_pred,smooth=1):
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])

    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def Dice_loss(y_true, y_pred):

    return 1-Dice(y_true, y_pred)


def weighted_categorical_crossentropy():
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy()
        model.compile(loss=loss,optimizer='adam')
    """
    weights = np.array([0.5,10])
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


########################## Test metric and loss ##########################


def dice_test(y_true, y_pred):

    smooth = 1

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.dot(y_true, K.transpose(y_pred))
    union = K.dot(y_true,K.transpose(y_true))+K.dot(y_pred,K.transpose(y_pred))
    return (2. * intersection + smooth) / (union + smooth)


def test_loss(y_true,y_pred):
    return K.mean(1-dice_test(y_true, y_pred),axis=-1)


def F1_score(y_true, y_pred):
    
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    
    return K.mean(f1) 


def matthews_correlation(y_true, y_pred):
    
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


########################## Plot functions ##########################


def get_history(metric,data_dict=None,history=None,initialize=False):
    """Save loss and acc for a given batch. Update a dict that contain all of these values
    Used only in test_one_model function (because of the ImageDataGenerator.flow method)   

    Args:
    metric (str): Name of metric used 
    data_dict (None or dictionnary): None if initialize set to True, otherwis the dictionnary to be updated
    history (History, keras object): contain a dictionnary that has loss / val loss & acc / val acc for a given minibatch run
    initialize (bool): If set to True, initialize the dictionnary that will contain the loss & acc values

    Returns:
    dictionnary: empty if initialize set to True, otherwise the updated dict of acc & loss per mini batch run
    """

    val_metric = 'val_' + metric
    
    if initialize: 

        # We create and fill dictionnary with loss and acc values 
        data_dict = {}
        data_dict['loss'] = []
        data_dict['val_loss'] = []
        data_dict[metric] = []
        data_dict[val_metric] = []
        
        return data_dict

    else: 

        # We append the loss & acc value for this batch to the dict (because there is one epoch per batch, the history dict has only one elem per list)
        data_dict['loss'].append(history.history['loss'][0])
        data_dict['val_loss'].append(history.history['val_loss'][0])
        data_dict[metric].append(history.history[metric][0])
        data_dict[val_metric].append(history.history[val_metric][0])


def plot_model_perf(data_dict,metric,model,load_from_dict=False,print_val=True): 
    """Plot accuracy (dice score) and loss per epoch. Saves the data 

    Args:
    data_dict (dictionnary): contain a dictionnary that has loss / val loss & acc / val acc
    metric (str): name of metric (for retrieving data)
    model (str): name of model used (for saving purposes)
    load_from_dict (bool): Used when a problem occured and only the dictionnary of value was saved. 
    print_val (bool): Set to false when printing loss & acc per batch one one epoch. If true, then print a line f
    """

    # First save data (if problem occurs) and set variables
    if not load_from_dict: 
        path = '../figure/data_for_' + model
        np.savez_compressed(path,a=data_dict) 
        savename_acc = '../figure/acc_plot_' + model + '.png'
        savename_loss = '../figure/loss_plot_' + model + '.png'

    else: 
    
        savename_acc = 'C:/Users/mbachelo/Desktop/Curie/Code/Cassandre/Figures/acc_plot_' + model + '.png'
        savename_loss = 'C:/Users/mbachelo/Desktop/Curie/Code/Cassandre/Figures/loss_plot_' + model + '.png'


    # creating plot variables 
    length = len(data_dict[metric])
    epochs = [i+1 for i in range(length)]
    val_metric = 'val_' + metric
    max_acc = np.ones(length,dtype=int) 
    min_loss = np.zeros(length,dtype=int)

    # Plot acc
    plt.figure()
    plt.plot(epochs,data_dict[metric], color='b')
    if print_val: 
        plt.plot(epochs,data_dict[val_metric], color='r')

    # Plot a ref line to see if we got an accuracy above 1
    plt.plot(epochs,max_acc,color='g',linestyle='dashed')

    plt.title('model accuracy')
    plt.ylabel('accuracy ({0})'.format(metric))
    plt.xlabel('epoch')
    if print_val: 
        plt.legend(['train', 'test','Max acc'], loc='best')
    else: 
        plt.legend(['Train accuracy on batch','Max acc'], loc='best')

    plt.savefig(savename_acc)
    plt.close()

    # Plot loss
    plt.figure() 
    plt.plot(epochs,data_dict['loss'], color='b')
    if print_val: 
        plt.plot(epochs,data_dict['val_loss'], color='r')

    # Plot a ref line to see if we got a loss below 0
    plt.plot(epochs,min_loss,color='g',linestyle='dashed')

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    if print_val: 
        plt.legend(['train', 'test', 'Min loss'], loc='best')
    else: 
        plt.legend(['Train loss on batch','Min loss'], loc='best') 

    plt.savefig(savename_loss)
    print('Done')


def plot_model_test(metric,model):
    """Plot all accuracy &  loss on a 

    Args:
    metric (str): name of metric (for retrieving data)
    model (str): name of model used (for saving purposes)

    """

    path_data = 'C:/Users/mbachelo/Desktop/Curie/Code/Cassandre/Figures/'
    opt_dict = ['_Adadelta_lr_1.0.npz','_Adagrad_lr_0.01.npz','_Nadam_lr_0.002.npz','_Adam_lr_0.001.npz']
    prefix = 'data_for_{}'.format(model)
    path_list = [path_data + prefix + opt_dict[i] for i in range(4)]

    # Load data dict
    data_dict_Adadelta = np.load(path_list[0])['a'].reshape(1)[0]
    data_dict_Adagrad = np.load(path_list[1])['a'].reshape(1)[0]
    data_dict_Nadam = np.load(path_list[2])['a'].reshape(1)[0]
    data_dict_Adam = np.load(path_list[3])['a'].reshape(1)[0]

    savename_acc = 'C:/Users/mbachelo/Desktop/Curie/Code/Cassandre/Figures/acc_plot_' + model + '.png'
    savename_loss = 'C:/Users/mbachelo/Desktop/Curie/Code/Cassandre/Figures/loss_plot_' + model + '.png'

    # creating plot variables 
    length = len(data_dict_Adadelta[metric])
    epochs = [i+1 for i in range(length)]
    val_metric = 'val_' + metric
    max_acc = np.ones(length,dtype=int) 
    min_loss = np.zeros(length,dtype=int)

    fig, axs = plt.subplots(1,2, constrained_layout=True, figsize=(15, 7))
    
    # Plot acc
    axs[0].plot(epochs,data_dict_Adadelta[metric], color='b')
    axs[0].plot(epochs,data_dict_Adagrad[metric], color='r')
    axs[0].plot(epochs,data_dict_Nadam[metric], color='c')
    axs[0].plot(epochs,data_dict_Adam[metric], color='y')
    axs[0].plot(epochs,max_acc,color='g',linestyle='dashed')
    axs[0].set_title('Model train {0} score'.format(metric))
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('accuracy ({0})'.format(metric))
    
    axs[1].plot(epochs,data_dict_Adadelta[val_metric], color='b')
    axs[1].plot(epochs,data_dict_Adagrad[val_metric], color='r')
    axs[1].plot(epochs,data_dict_Nadam[val_metric], color='c')
    axs[1].plot(epochs,data_dict_Adam[val_metric], color='y')
    axs[1].plot(epochs,max_acc,color='g',linestyle='dashed')
    axs[1].set_title('Model val {0} score'.format(metric))
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('accuracy ({0})'.format(metric))

    fig.legend(['Adadelta', 'Adagrad', 'Nadam', 'Adam', 'Max acc'], bbox_to_anchor=(0.99, 0.8))
    fig.suptitle('Accuracy for {0} with different optimizers'.format(model), fontsize=16)
    plt.savefig(savename_acc)
    plt.close()

    # Plot loss
    fig, axs = plt.subplots(1,2, constrained_layout=True, figsize=(15, 7))

    axs[0].plot(epochs,data_dict_Adadelta['loss'], color='b')
    axs[0].plot(epochs,data_dict_Adagrad['loss'], color='r')
    axs[0].plot(epochs,data_dict_Nadam['loss'], color='c')
    axs[0].plot(epochs,data_dict_Adam['loss'], color='y')
    axs[0].plot(epochs,min_loss,color='g',linestyle='dashed')
    axs[0].set_title('Model train loss (binary crossentropy)')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    
    axs[1].plot(epochs,data_dict_Adadelta['val_loss'], color='b')
    axs[1].plot(epochs,data_dict_Adagrad['val_loss'], color='r')
    axs[1].plot(epochs,data_dict_Nadam['val_loss'], color='c')
    axs[1].plot(epochs,data_dict_Adam['val_loss'], color='y')
    axs[1].plot(epochs,min_loss,color='g',linestyle='dashed')
    axs[1].set_title('Model val loss (binary crossentropy)')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('loss')

    fig.legend(['Adadelta', 'Adagrad', 'Nadam', 'Adam', 'Max acc'], bbox_to_anchor=(0.99, 0.3))
    fig.suptitle('Loss for {0} with different optimizers'.format(model), fontsize=16)
    plt.savefig(savename_loss)
    plt.close()


########################## Model generator help functions ##########################


def get_dim_for_model(X_train,dimension,just_shape=False):
    """Return the input tuple for input shape in the NN

    Args:
    X_train (np.ndarray or tuple): Array to be fed in the NN or just the array'shape 
    dimension (bool): Specifie dimension (the input shape will change in the case of 3D segmentation)
    just_shape (bool): Specify if the X_train input is a tuple or an ndarray

    """

    if not just_shape:     
        if dimension: 
            feature = 1
            x, y, frame = X_train.shape[2], X_train.shape[2], X_train.shape[1]
            input_img = (frame, x, y, feature)
        else:
            inChannel = 1
            x, y = X_train.shape[1], X_train.shape[1]
            input_img = (x, y, inChannel)
    else: 
        if dimension: 
            feature = 1
            x, y, frame = X_train[2], X_train[2], X_train[1]
            input_img = (frame, x, y, feature)
        else:
            inChannel = 1
            x, y = X_train[1], X_train[1]
            input_img = (x, y, inChannel)
    
    return input_img 


def test_all_models(list_model,metric): 

    path = 'C:/Users/mbachelo/Desktop/Curie/Code/Cassandre/Figures/Test_{}'.format(metric)

    data_dict_acc = {}
    data_dict_loss = {}
    length = 0
    
    for i, model in enumerate(list_model):

        model_loss = np.load(os.path.join(path,'data_batch_epoch_0_for_{}.npz'.format(model)))['a'].tolist()
        model_acc = np.load(os.path.join(path,'data_batch_epoch_0_for_{}.npz'.format(model)))['b'].tolist()
        data_dict_acc[model] = model_acc
        data_dict_loss[model] = model_loss 
        length = len(model_acc)

    # data_dict_acc['Max Acc'] = [1]*length
    data_dict_loss['Min loss'] = [0]*length


    acc_df = pd.DataFrame(data_dict_acc)
    loss_df = pd.DataFrame(data_dict_loss)
    print(acc_df.head())
    print(loss_df.head())

    savename_acc = os.path.join(path,'Acc_summary.png')
    savename_loss = os.path.join(path,'Loss_summary.png')

    # Plotting loss
    sns.lineplot(data=loss_df)
    plt.xlabel('Number of patients')
    plt.ylabel('Weighted categorical crossentropy')
    plt.title('Model loss evolution over batch on one epoch')
    plt.savefig(savename_loss)
    plt.close()
    
    # Plotting acc
    sns.lineplot(data=acc_df)
    plt.xlabel('Number of patients')
    plt.ylabel('Sørensen & Dice coefficient')
    plt.title('Model accuracy evolution over batch on one epoch')
    plt.savefig(savename_acc)


def recreate_epoch_plot(model,number_of_epoch,metric,print_val):

    path = 'C:/Users/mbachelo/Desktop/Curie/Code/Cassandre/Figures/'
    list_loss, list_acc = [],[]

    for i in range(number_of_epoch):

        model_loss = np.load(os.path.join(path,'data_batch_epoch_{0}_for_{1}.npz'.format(i,model)))['a'].tolist()
        model_acc = np.load(os.path.join(path,'data_batch_epoch_{0}_for_{1}.npz'.format(i,model)))['b'].tolist()
        list_loss += model_loss
        list_acc += model_acc

    if print_val: 
        
        data_val = np.load(os.path.join(path,'data_for_{}.npz'.format(model)))['a'].reshape(1)[0]
        val_metric = 'val_' + metric
        print(data_val)
        
        for i in range(number_of_epoch):
            
            if i == 0:
                
                list_val_loss = [data_val['val_loss'][i]]*72
                list_val_acc = [data_val[val_metric][i]]*72

            else: 

                list_val_loss += [data_val['val_loss'][i]]*72
                list_val_acc += [data_val[val_metric][i]]*72

        data_dict = {'loss':list_loss,metric:list_acc,'val_loss':list_val_loss,val_metric:list_val_acc}
        plot_model_perf(data_dict,metric,model,True,True)

    else: 
        data_dict = {'loss':list_loss,metric:list_acc}
        plot_model_perf(data_dict,metric,model,True,False)


def plot_preds(mri,y_true,y_pred,model_str,function_metric,index_list=None,random_sample=False,sample_size=None):

    # Prevent error 
    if random_sample == True and sample_size == None:
        raise ValueError('For random prediction plot you need to provide the sample size (positive integers)')
    if random_sample == False and index_list == None:  
        raise ValueError('For specific prediction plot you need to provide the index list (list of integers)')

    # In the case of random prediction we pick random frame amongs those provided
    if random_sample: 
        list_of_index = rd.sample(range(0,len(y_pred)),sample_size)
        print('Choosen Sample : {}\n'.format(list_of_index))
    else: 
        sample_size = len(index_list)

    # Create figure 
    plt.figure(figsize=(2*sample_size, 2*sample_size))
    
    for i,sample in enumerate(list_of_index):

            current_pred = get_frame(0,y_pred,sample)
            current_X = get_frame(0,mri,sample)
            current_truth = get_frame(0,y_true,sample)

            print(current_truth.shape,current_pred.shape,current_X.shape)

            plt.subplot(3,sample_size,i+1)
            plt.imshow(current_X.T, cmap="gray", origin="lower")
            plt.subplot(3,sample_size,i+1+sample_size)
            plt.imshow(current_pred.T, cmap="gray", origin="lower")
            plt.subplot(3,sample_size,i+1+2*sample_size)
            plt.imshow(current_truth.T, cmap="gray", origin="lower")

    plt.suptitle("Prediction on test data")
    plt.savefig('../figure/plot_preds_for_{0}_{1}.png'.format(model_str,function_metric))


def plot_histogram(img): 

    # Check voxel distribution
    mu, sigma = np.mean(img), np.std(img)
    count, bins, ignored = plt.hist(img, 30, density=True) 
    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    
    return bins,mu,sigma


if __name__ == '__main__':

    list_model  = ['UNet','PSPnet']
    # recreate_epoch_plot('UNet',2,'matthews_correlation',True)
    
    for model in list_model: 

        plot_model_test('matthews_correlation',model)