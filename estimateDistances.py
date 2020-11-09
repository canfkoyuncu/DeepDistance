##################################################################################
# This python code includes the function call to estimate the distance maps
# using a trained DeepDistanceModel or DeepDistanceExtendedModel on a given
# test set 'x' and returns the estimated output maps.
# When extendedModel = False, it uses DeepDistanceModel and returns an empty
# list for the segmentation output.
##################################################################################


import numpy as np
from keras.models import load_model


def test(modelpath = '', x = [],         
         extendedModel = False, # is DeepDistanceExtendedModel used
         patch_height=512, patch_width=512, increment = 64 ): 

    segmentationMaps = [] # result
    innerDstMaps = [] # result
    outerDstMaps = [] # result
    
    model = load_model(modelpath)
    
    for ind in range(len(x)): #index for each image
            print ('img ind: %d / %d' % (ind+1, len(x)))
            shape = (x[ind].shape[0],x[ind].shape[1])
            
            segmentation = np.zeros(shape)
            inner_dst = np.zeros(shape)
            outer_dst = np.zeros(shape)
            counts = np.zeros(shape)
            
            i=0    

            while (i < x[ind].shape[0] +1): 
                j=0
                while (j < x[ind].shape[1] +1): 

                    
                    if( i < (x[ind].shape[0] - patch_height +1) and j < (x[ind].shape[1] - patch_width +1)):
                        x_start = i
                        x_end = i+patch_height
                        y_start = j
                        y_end = j+patch_width
                    else:
                        if(i < (x[ind].shape[0] - patch_height +1)):
                            x_start = i
                            x_end = i+patch_height
                            y_start = x[ind].shape[1] - patch_width
                            y_end = x[ind].shape[1]        
                            j = x[ind].shape[1] +1

                        elif(j < (x[ind].shape[1] - patch_width +1)):
                            x_start = x[ind].shape[0] - patch_height
                            x_end = x[ind].shape[0]
                            y_start = j
                            y_end = j+patch_width
                            i = x[ind].shape[0] +1
                        else:
                            x_start = x[ind].shape[0] - patch_height
                            x_end = x[ind].shape[0]
                            y_start = x[ind].shape[1] - patch_width
                            y_end = x[ind].shape[1]
                            i = x[ind].shape[0] +1
                            j = x[ind].shape[1] +1

                    counts[x_start:x_end , y_start:y_end] += 1
                    patch = x[ind][x_start:x_end , y_start:y_end , :]

                    #Model - dist
                    pred = model.predict( patch.reshape( (1,) + patch.shape )  )
                    
                    if(extendedModel):
                        segmentation[x_start:x_end , y_start:y_end] += pred[0][0,:,:,0]            
                        inner_dst[x_start:x_end , y_start:y_end] += pred[1][0,:,:,0]
                        outer_dst[x_start:x_end , y_start:y_end] += pred[2][0,:,:,0] 
                             
                    else:
                        inner_dst[x_start:x_end , y_start:y_end] += pred[0][0,:,:,0]  
                        outer_dst[x_start:x_end , y_start:y_end] += pred[1][0,:,:,0] 
                
                    j+= increment
                i += increment
            

            if(extendedModel):
                mask = np.divide(segmentation,counts)  #average              
                segmentationMaps.append(mask)
                
            inner_dstmap = np.divide(inner_dst,counts) #average
            innerDstMaps.append(inner_dstmap)
            outer_dstmap = np.divide(outer_dst,counts) #average
            outerDstMaps.append(outer_dstmap)

    return[innerDstMaps, outerDstMaps, segmentationMaps]
    
    
    
    
