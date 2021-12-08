########################################################## READ ME ##########################################################
#                                                                                                                           #
#                                                Custom image data generator                                                #
#                           It loads X and Y images stored in train val & test folders as npz files                         #                              
#                                                                                                                           #                              
#############################################################################################################################


########################## Imports ##########################


import numpy as np
import keras
import os
from utils import * 
from image_processing import * 


########################## Generator class ##########################


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, sampling_data, train_val_test, cut_axis, operator, shuffle=True):
        # Initialization
        self.shuffle = shuffle
        self.operator = operator
        self.cut_axis = cut_axis
        self.sampling_data = sampling_data
        self.train_val_test = train_val_test
        self.choice = self.choose_between_train_val_and_test()
        self.full_path = str(self.sampling_data[self.choice][1])
        self.on_epoch_end()

    def __len__(self):
        """Take all batches in each iteration"""
        
        # The first element of sampling data is the total number of patients in the directory. that's the length we need to return
        return int(self.sampling_data[self.choice][0])

    def __getitem__(self,index):
        """Generate one batch of data"""
        
        # Generate indexes of the batch corresponding to a single file
        indexes = self.indexes[index:(index+1)]    
        # Generate data
        X, y = self.__data_generation()
        
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
    
        number_of_patients = int(self.sampling_data[self.choice][0])
        self.indexes = np.arange(number_of_patients)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def choose_between_train_val_and_test(self):
        """Return a index which will be used in sampling_data to have the correct path to data"""
        
        if self.train_val_test == 'train':  
            return 0 
        elif self.train_val_test == 'val': 
            return 1 
        elif self.train_val_test == 'test': 
            return 2
        else: 
            raise ValueError('Expected train, val or test as argument (string). Got {0} (type {1}) as input instead'.format(self.train_val_test,type(self.train_val_test)))

    def __data_generation(self):
        """Generates data containing batch_size samples""" 
    
        cut_name = name_cut(self.cut_axis)

        # Get to the given patients 
        for patients in self.indexes: 
            patient_name = 'patient_' + str(patients+1) + '_op_' + str(self.operator) + cut_name + '.npz'
            name_X = os.path.join(self.full_path, 'MRI', patient_name)
            name_Y = os.path.join(self.full_path, 'Segmentation', patient_name)
            X = np.load(name_X)['a']
            Y = np.load(name_Y)['a']

            # We set X vectors values between 0 and 1 + normalize it 
            X = X_train_transform(X)

        return X,Y