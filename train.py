###############################################################################################
# This python code includes the function calls to train the proposed
# DeepDistanceModel and DeepDistanceExtendedModel. This file also includes the
# function to prepare training/validation patches from a set of training images.
## 'patchdata' dictionary should contain the training data as follows:
#     for training set   : 'x_patches', 'inner_dst_patches', 'outer_dst_patches'
#     for validation set : 'x_valid_patches', 'inner_dst_valid_patches', 'outer_dst_valid_patches'
## 'modelpath' should be the desired directory to save the model.
###############################################################################################

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from deepDistanceModels import deepDistanceModel, deepDistanceExtendedModel

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Use gpu-0 only
    
###############################################################################################

def trainDeepDistanceModel(patchdata, modelpath = './deepDistance.hdf5'):
    
    model =  deepDistanceModel(input_height=512, input_width=512)
    model.summary()
    print('Model is ready !')
    
    
    checkpointer = ModelCheckpoint(modelpath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    earlystopper = EarlyStopping(patience=100, verbose=0)
    
    
    hist = model.fit(x = patchdata['x_patches'],
                     y= [patchdata['inner_dst_patches'], patchdata['outer_dst_patches']],
                     validation_data = (patchdata['x_valid_patches'], [ patchdata['inner_dst_valid_patches'], patchdata['outer_dst_valid_patches'] ] ),
                     epochs=300, batch_size=1,
                     shuffle=True, validation_split=0, callbacks=[checkpointer,earlystopper])
    
    print('Model is trained !')
    

def trainDeepDistanceExtendedModel(patchdata, modelpath = './deepDistanceExtended.hdf5'):
    
    model =  deepDistanceExtendedModel(input_height=512, input_width=512)
    model.summary()
    print('Model is ready !')
    
    
    checkpointer = ModelCheckpoint(modelpath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    earlystopper = EarlyStopping(patience=100, verbose=0)
    
    
    hist = model.fit(x = patchdata['x_patches'], 
                     y= [patchdata['y_patches'], patchdata['inner_dst_patches'], patchdata['outer_dst_patches']],
                     validation_data = (patchdata['x_valid_patches'], [ patchdata['y_valid_patches'], patchdata['inner_dst_valid_patches'], patchdata['outer_dst_valid_patches'] ] ),
                     epochs=300, batch_size=1,
                     shuffle=True, validation_split=0, callbacks=[checkpointer,earlystopper])
    
    print('Model is trained !')
        

###############################################################################################
def normalizeImg(img):
    norm_img = np.zeros(img.shape)
    for i in range(3):
        norm_img[:,:,i] = (img[:,:,i] - img[:,:,i].mean()) / (img[:,:,i].std())
    return norm_img

###############################################################################################
# x_data            : The array of input images
# inner_dst_data    : The array of the corresponding inner distance maps for the input images
# outer_dst_data    : The array of the corresponding normalized outer distance maps for the
#                     input images
# y_data            : The array of segmentation maps (it is necessary only if the extended
#                     version of the DeepDistance model is used)
#
# The return values of this function can be used to create a dictionary of patches used in
# the trainDeepDistanceModel function.
###############################################################################################
def crop_patches( x_data, inner_dst_data, outer_dst_data, y_data, 
                    patch_height=512, patch_width=512):
    
    i_increment = int(patch_height/2)
    j_increment = int(patch_width/2)
    #training patches
    x_patches = list()
    inner_dst_patches = list()
    outer_dst_patches = list()
    y_patches = list()

    for ind in range(x_data.shape[0]):
        img = x_data[ind,:,:,:]
        inner_dst = inner_dst_data[ind]
        outer_dst = outer_dst_data[ind]
        y = y_data[ind,:,:]
        i = 0    
        while(i + i_increment < img.shape[0]):
            j = 0
            while(j + j_increment  < img.shape[1]):
                
                if(i +patch_height < img.shape[0] and j +patch_width < img.shape[1]): # normal crop
                    height_start = i
                    height_end = i+patch_height
                    width_start = j
                    width_end = j+patch_width
                    
                elif(j +patch_width < img.shape[1]): # img boundary, crop last possible patch
                    height_end = img.shape[0]
                    height_start = height_end - patch_height     
                    width_end = j+patch_width
                    width_start = j
                    
                elif(i +patch_height < img.shape[0]):  # img boundary, crop last possible patch
                    height_start = i
                    height_end = i+patch_height
                    width_end = img.shape[1]
                    width_start = width_end - patch_width     

                else: # image corner
                    height_end = img.shape[0]
                    height_start = height_end - patch_height   
                    width_end = img.shape[1]
                    width_start = width_end - patch_width   
                         
                
                x_patches.append(img[height_start:height_end, width_start:width_end,:])
                inner_dst_patches.append(inner_dst[height_start:height_end, width_start:width_end])
                outer_dst_patches.append(outer_dst[height_start:height_end, width_start:width_end])
                y_patches.append(y[height_start:height_end, width_start:width_end])
                
                j += j_increment
                
            i += i_increment
                
    x_patches = np.asarray(x_patches)   
    inner_dst_patches = np.asarray(inner_dst_patches)
    outer_dst_patches  = np.asarray(outer_dst_patches)
    y_patches = np.asarray(y_patches)   

    return [x_patches, inner_dst_patches, outer_dst_patches, y_patches]
    


