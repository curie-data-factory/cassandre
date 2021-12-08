########################################################## READ ME ##########################################################
#                                                                                                                           #
#                   The different models that will be used in tumor segmentation are all defined in this script             #
#                       The purpose is to compare the efficiency of all of the different models listed below                #
#                                                                                                                           #
#############################################################################################################################

########################## Imports ##########################


from __future__ import print_function

import numpy as np
import warnings
import tensorflow as tf 

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.input_layer import Input 
from keras import layers

from utils import *


########################## Choosing optimizer ##########################


def building_optimizer(name,learning_rate,more):

    if more: 
            print('\n==========================================================================================================================================================')
            print('You chose to modify every single parameters for {0} optimizer'.format(name))
    else: 
        pass

    if name == 'RMSprop':
        if more: 
            rho = float(input('Enter rho value (must be > 0)'))
            epsilon = float(input('Enter epsilon (fuzz factor) value (must be >= 0)'))          
            decay = float(input('Enter decay value (must be >= 0)'))
        else:
            rho,decay,epsilon = 0.9,0.0,None
        
        print('Choosen hyperparameters:\n Rho {0} epsilon {1} decay {2}'.format(rho,epsilon,decay))
        return RMSprop(lr=learning_rate, rho=rho, epsilon=epsilon, decay=decay)

    elif name == 'Adam':
        if more: 
            beta_1 = float(input('Enter beta_1 value (must be between 0 and 1)'))
            beta_2 = float(input('Enter beta_2 value (must be between 0 and 1)'))
            epsilon = float(input('Enter epsilon (fuzz factor) value (must be >= 0)')) 
            decay = float(input('Enter decay value (must be >= 0)'))
            amsgrad = bool(input('Bool Whether to apply the AMSGrad variant (press y for yes, enter for no)'))
        else:
            beta_1,beta_2,decay,epsilon,amsgrad = 0.9,0.999,0.0,None,False
        
        print('Choosen hyperparameters:\n beta_1 {0} beta_2 {1} decay {2} epsilon {3} amsgrad {4}'.format(beta_1,beta_2,decay,epsilon,amsgrad))
        return Adam(lr=learning_rate,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,amsgrad=amsgrad)

    elif name == 'Nadam':
        if more: 
            beta_1 = float(input('Enter beta_1 value (must be between 0 and 1)'))
            beta_2 = float(input('Enter beta_2 value (must be between 0 and 1)'))
            epsilon = float(input('Enter epsilon (fuzz factor) value (must be >= 0)')) 
            decay = float(input('Enter decay value (must be >= 0)'))
            schedule_decay = float(input('Enter decay value (must be >= 0)'))
        else:
            beta_1,beta_2,decay,epsilon,schedule_decay = 0.9,0.999,0.0,None,0.004
        
        print('Choosen hyperparameters:\n beta_1 {0} beta_2 {1} decay {2} epsilon {3} schedule_decay {4}'.format(beta_1,beta_2,decay,epsilon,schedule_decay))
        return Nadam(lr=learning_rate,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,schedule_decay=schedule_decay)

    elif name == 'Adadelta':
        if more: 
            rho = float(input('Enter rho value (must be > 0)'))
            epsilon = float(input('Enter epsilon (fuzz factor) value (must be >= 0)'))          
            decay = float(input('Enter decay value (must be >= 0)'))
        else:
            rho,decay,epsilon = 0.95,0.0,None
        
        print('Choosen hyperparameters:\n Rho {0} epsilon {1} decay {2}'.format(rho,epsilon,decay))
        return Adadelta(lr=learning_rate, rho=rho, epsilon=epsilon, decay=decay)

    elif name == 'Adagrad':
        if more: 
            epsilon = float(input('Enter epsilon (fuzz factor) value (must be >= 0)'))          
            decay = float(input('Enter decay value (must be >= 0)'))
        else:
            decay,epsilon = 0.0,None
        
        print('Choosen hyperparameters:\n epsilon {0} decay {1}'.format(epsilon,decay))
        return Adagrad(lr=learning_rate, epsilon=epsilon, decay=decay)

    else: 
        if more: 
            momentum = float(input('Enter momentum value (must be > 0)'))
            nesterov = bool(input('Bool Whether to apply the Nesterov momentum (just press enter if you dont want to, otherwise type y)'))          
            decay = float(input('Enter decay value (must be >= 0)'))
        else:
            momentum,nesterov,decay = 0.0,False,0.0

        print('Choosen hyperparameters:\n momentum {0} nesterov {1} decay {2}'.format(momentum,nesterov,decay))
        return SGD(lr=learning_rate,momentum=momentum,nesterov=nesterov)

########################## Auto-encoder ##########################


def AutoEncoder(inp, classes, input_optimizer=None, more=False):
    
    # Input and model set-up
    inp = Input(inp)
    inp = BatchNormalization()(inp)

    # Encoder 
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inp) 
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1) 
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2) 
    
    # Code 
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2) 
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) 
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    up1 = UpSampling2D((2,2))(conv4) 
    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) 
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    up2 = UpSampling2D((2,2))(conv5) 
    
    # Last layer
    decoded = Conv2D(classes, (3, 3), activation='sigmoid', padding='same')(up2) 
    
    # Model building and choosing optimizer
    if input_optimizer == None: 
        optimizer = RMSprop()
    else: 
        name, learning_rate = input_optimizer[0],input_optimizer[1]
        optimizer = building_optimizer(name,learning_rate,more)

    model = Model(inputs=[inp], outputs=[decoded])
    loss = weighted_categorical_crossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[F1_score])

    # Print info on the model 
    print(model.summary())
    metric = 'F1_score'
    string = 'Chosen Metric : ' + metric
    centered = string.center(60)
    print('\n',centered,'\n')
    
    return model,metric


def AutoEncoder3D(inp, classes, input_optimizer=None, more=False):
    
    # Input and model set-up
    inp = Input(inp)
    inp = BatchNormalization()(inp)

    # encoder 
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inp) 
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1) 
    
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2) 
    
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2) 
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)

    #decoder
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3) 
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv4)
    upcv1 = UpSampling3D((2, 2, 2))(conv4) 
    
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(upcv1) 
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv5)
    upcv2 = UpSampling3D((2, 2, 2))(conv5) 

    # Last layer
    lastL = Conv3D(classes, (3, 3, 3), activation='sigmoid', padding='same')(upcv2) 

    # Model building and choosing optimizer
    if input_optimizer == None: 
        optimizer = RMSprop()
    else: 
        name, learning_rate = input_optimizer[0],input_optimizer[1]
        optimizer = building_optimizer(name,learning_rate,more)

    model = Model(inputs=[inp], outputs=[lastL])
    model.compile(optimizer=optimizer, loss=Dice_loss, metrics=[Dice])

    # Print info on the model 
    print(model.summary())
    metric = 'Dice'
    string = 'Chosen Metric : ' + metric
    centered = string.center(60)
    print('\n',centered,'\n')

    return model,metric


########################## U-Net ##########################


def UNet(inp, classes, input_optimizer=None, more=False):

    # Input and model set-up
    inp = Input(inp)

    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inp)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D((2, 2)) (c4)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    # Last block
    outputs = Conv2D(classes, (1, 1), activation='sigmoid') (c9)

    # Model building and choosing optimizer
    if input_optimizer == None: 
        optimizer = Adadelta()
    else: 
        name, learning_rate = input_optimizer[0],input_optimizer[1]
        optimizer = building_optimizer(name,learning_rate,more)

    # Because class are imbalanced, we need to set weigths to tumor's class 
    model = Model(inputs=[inp], outputs=[outputs])
    loss = weighted_categorical_crossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[matthews_correlation])

    # Print info on the model 
    print(model.summary())
    metric = 'matthews_correlation'
    string = 'Chosen Metric : ' + metric
    centered = string.center(90)
    print('\n',centered,'\n')

    return model,metric 


########################## PSP net ##########################


def resize_image_in_network(inp,s):

    data_format = 'channels_last'  
    return Lambda(lambda x: K.resize_images(x, height_factor=s[0], width_factor=s[1], data_format=data_format, interpolation='bilinear'))(inp)


def PSPnet(input_shape, classes, input_optimizer=None, more=False):

    inputs = Input(input_shape)

    x = Conv2D(64, (3, 3), padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='act1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='act2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)

    for i in range(3):
        x = Conv2D(256, (3, 3), padding='same', name='conv'+str(i+3))(x)
        x = BatchNormalization(name='bn'+str(i+3))(x)
        x = Activation('relu', name='act'+str(i+3))(x)
        x = MaxPooling2D((2, 2), name='pool'+str(i+3))(x)

    # At this point, with an input shape of (416,416), the last x has a shape of (13,13)
    # Since 13 is a prime number, we have to resample it into a higher dim image in order to create the pyramids 
    residual = resize_image_in_network(x,(4,4))   
    
    # Pyramid pooling part  
    output_pool = []
    pool_factors = [1,2,4,13]
    stride_list = [(i,i) for i in pool_factors]

    for i,size in enumerate(pool_factors): 

        # Creating pyramid of different size (determined by the pooling factor variable size)
        stride = stride_list[i]
        x = AveragePooling2D(stride, strides=stride, padding='same', name='avg_pool'+str(i+1))(residual)
        x = Conv2D(512, (1, 1), padding='same', use_bias=False, name='conv'+str(i+6))(x)
        x = BatchNormalization(name='bn'+str(i+6))(x)
        x = Activation('relu',name='act'+str(i+6))(x)
        x = resize_image_in_network(x,stride)
        output_pool.append(x)

    # Assembling block: concatenate all pyramid results
    x = Concatenate(axis=-1)(output_pool)
    x = Conv2D(512, (1, 1), use_bias=False, name='Final_conv')(x)
    x = BatchNormalization(name='Final_bn')(x)
    x = Activation('relu', name='Final_act')(x)

    # Last block. We use a Conv2DTranspose in order to have same image dimension as input image
    x = Conv2D(classes, (3, 3), padding='same', name='output_conv')(x)
    x = Conv2DTranspose(classes, kernel_size=(64, 64), strides=(8, 8), activation='sigmoid', padding='same')(x)

    # Model building and choosing optimizer
    if input_optimizer == None: 
        optimizer = Adam()
    else: 
        name, learning_rate = input_optimizer[0],input_optimizer[1]
        optimizer = building_optimizer(name,learning_rate,more)

    model = Model(input=inputs, output=x)
    loss = weighted_categorical_crossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[matthews_correlation])

    # Print info on the model 
    print(model.summary())
    metric = 'matthews_correlation'
    string = 'Chosen Metric : ' + metric
    centered = string.center(90)
    print('\n',centered,'\n')

    return model,metric 

    
########################## VGG16 ##########################


def VGG16(input_shape, classes, input_optimizer=None, more=False):

    inputs = Input(input_shape)

    x = BatchNormalization()(inputs)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding="same")(x)
    x = Conv2DTranspose(classes, kernel_size=(64, 64), strides=(32, 32), activation='linear', padding='same')(x)

    # Model building and choosing optimizer
    if input_optimizer == None: 
        optimizer = Adadelta()
    else: 
        name, learning_rate = input_optimizer[0],input_optimizer[1]
        optimizer = building_optimizer(name,learning_rate,more)

    # sigmoid
    reshape = Reshape((-1,1))(x)
    act = Activation('sigmoid')(reshape)

    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])

    # Print info on the model
    print(model.summary()) 
    metric = 'acc'    
    string = 'Chosen Metric : ' + metric
    centered = string.center(60)
    print('\n',centered,'\n')

    return model,metric 


def VGG16_3D(input_shape, classes, input_optimizer=None, more=False):

    inputs = Input(input_shape)

    x = BatchNormalization()(inputs)

    # Block 1
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv3D(512, (3, 3), 3, activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(x)

    x = Conv3D(512, (3, 3, 3), activation='relu', padding="same")(x)
    x = Conv3DTranspose(classes, kernel_size=(64, 64, 64), strides=(32, 32, 32), activation='linear', padding='same')(x)

    # Model building and choosing optimizer
    if input_optimizer == None: 
        optimizer = Adadelta()
    else: 
        name, learning_rate = input_optimizer[0],input_optimizer[1]
        optimizer = building_optimizer(name,learning_rate,more)

    # sigmoid
    reshape= Reshape((-1,1))(x)
    act = Activation('sigmoid')(reshape)

    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metric=['accuracy'])

    # Print info on the model
    print(model.summary()) 
    metric = 'acc'
    string = 'Chosen Metric : ' + metric
    centered = string.center(60)
    print('\n',centered,'\n')

    return model,metric


########################## Xception ##########################


def Xception(input_shape, classes, input_optimizer=None, more=False):
    
    inputs = Input(input_shape)
    include_top=True

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(inputs)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Flatten(name='Flatten')(x)
        x = Dense(1024, activation='relu', name='fully_connected')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else: 
        x = UpSampling2D(size=(32,32))(x)
        x = Conv2D(classes, 3, activation='sigmoid', padding='same')(x)

    # Model building and choosing optimizer
    if input_optimizer == None: 
        optimizer = Adam()
    else: 
        name, learning_rate = input_optimizer[0],input_optimizer[1]
        optimizer = building_optimizer(name,learning_rate,more)

    model = Model(input=inputs, output=x)
    loss = weighted_categorical_crossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Print info on the model 
    print(model.summary())
    metric = 'Sorensen_Dice'
    string = 'Chosen Metric : ' + metric
    centered = string.center(90)
    print('\n',centered,'\n')

    return model,metric 


########################## SegNet ##########################


def SegNet(input_shape,classes,input_optimizer=None, more=False):

    inputs = Input(input_shape)
    x = BatchNormalization()(inputs)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Up Block 1
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 2
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 3
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 4
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 5
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Last layer
    x = Conv2D(classes, (1, 1), activation='sigmoid', padding='same')(x)

    # Model building and choosing optimizer
    if input_optimizer == None: 
        optimizer = Adam()
    else: 
        name, learning_rate = input_optimizer[0],input_optimizer[1]
        optimizer = building_optimizer(name,learning_rate,more)

    model = Model(input=inputs, output=x)
    loss = weighted_categorical_crossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Print info on the model 
    print(model.summary())
    metric = 'acc'
    string = 'Chosen Metric : ' + metric
    centered = string.center(60)
    print('\n',centered,'\n')

    return model,metric

########################## Function for model selection ##########################


def choose_model_from_arg(arg_model,arg_dim,input_model,classes,input_optimizer=None,more=False):
    """Given different arguments, choose the right model
    
    Args:
    arg_model (str): model architecture name between the following one: ['AE','PSPnet','UNet','VGG16','Xception','SegNet']
    arg_dim (bool): 2D (if False) or 3D (if True) model architecture
    input_model (tuple): shapes of the input for the model
    classes (int): Number of classes for prediction. Equals to 1 if weighted_map is True, 2 otherwise 
    input_optimizer (None or list): If None, the default optimizer for the choosen model will be used. 
    Otherwise the model will be compiled with the optimizer and learning rate provided in this argument  
    more (bool): If set to true, then the user will be asked to provide the value for all choosen optimizer hyperparameters

    Returns:
    a model class from keras 
    """

    arg_list = ['AE','PSPnet','UNet','VGG16','Xception','SegNet']

    # Print function for information 
    print('\n==========================================================================================================================================================')
    string = 'Chosen model : ' + arg_model
    
    if arg_dim:
        string += ' (3D)'
    else: 
        pass

    centered = string.center(152)
    print('\n',centered,'\n')
    print('==========================================================================================================================================================\n')


    for i in range(len(arg_list)):

        if arg_model == arg_list[0]: 
            if arg_dim: 
                return AutoEncoder3D(input_model,classes,input_optimizer,more)
            else: 
                return AutoEncoder(input_model,classes,input_optimizer,more)
        
        if arg_model == arg_list[1]: 
            if arg_dim: 
                return PSPnet3D(input_model,classes,input_optimizer,more)
            else: 
                return PSPnet(input_model,classes,input_optimizer,more)

        elif arg_model == arg_list[2]: 
            if arg_dim: 
                return UNet3D(input_model,classes,input_optimizer,more)
            else: 
                return UNet(input_model,classes,input_optimizer,more)
        
        elif arg_model == arg_list[3]: 
            if arg_dim: 
                return VGG16_3D(input_model,classes,input_optimizer,more)
            else: 
                return VGG16(input_model,classes,input_optimizer,more)

        elif arg_model == arg_list[4]: 
            return Xception(input_model,classes,input_optimizer,more)

        else: 
            if arg_dim: 
                return SegNet3D(input_model,classes,input_optimizer,more)
            else: 
                return SegNet(input_model,classes,input_optimizer,more)
 
        