
##############################################################################
# This python code includes the network architectures for the DeepDistance
# model and DeepDistanceExtended model. It also includes parameter settings
# used in these architectures.
##############################################################################

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras import optimizers

##############################################################################
def deepDistanceExtendedModel(input_height=512, input_width=512, nChannels = 3):

    optimizer = optimizers.Adadelta()
    inputs = Input(shape= (input_height, input_width, nChannels))

    #Encoder (Shared)
    conv1 = Convolution2D(64,(3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128,(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(128,(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(256,(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(256,(3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Convolution2D(512,(3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(512,(3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Convolution2D(1024,(3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(1024,(3, 3), activation='relu', padding='same')(conv5)

    #Decoder 1
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Convolution2D(512,(3, 3), activation='relu', padding='same')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv6)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Convolution2D(256,(3, 3), activation='relu', padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv7)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Convolution2D(128,(3, 3), activation='relu', padding='same')(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv8)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Convolution2D(64,(3, 3), activation='relu', padding='same')(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv9)
    
    out1 = Convolution2D(1, (1,1), activation = 'sigmoid')(conv9)

    #Decoder 2
    up1_ = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6_ = Convolution2D(512,(3, 3), activation='relu', padding='same')(up1_)
    conv6_ = Dropout(0.2)(conv6_)
    conv6_ = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv6_)
    
    up2_ = concatenate([UpSampling2D(size=(2, 2))(conv6_), conv3], axis=3)
    conv7_ = Convolution2D(256,(3, 3), activation='relu', padding='same')(up2_)
    conv7_ = Dropout(0.2)(conv7_)
    conv7_ = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv7_)
    
    up3_ = concatenate([UpSampling2D(size=(2, 2))(conv7_), conv2], axis=3)
    conv8_ = Convolution2D(128,(3, 3), activation='relu', padding='same')(up3_)
    conv8_ = Dropout(0.2)(conv8_)
    conv8_ = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv8_)
    
    up4_ = concatenate([UpSampling2D(size=(2, 2))(conv8_), conv1], axis=3)
    conv9_ = Convolution2D(64,(3, 3), activation='relu', padding='same')(up4_)
    conv9_ = Dropout(0.2)(conv9_)
    conv9_ = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv9_)
    
    out2 = Convolution2D(1, (1,1), activation = 'relu')(conv9_)

    
    #Decoder 3
    up1__ = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6__ = Convolution2D(512,(3, 3), activation='relu', padding='same')(up1__)
    conv6__ = Dropout(0.2)(conv6__)
    conv6__ = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv6__)
    
    up2__ = concatenate([UpSampling2D(size=(2, 2))(conv6__), conv3], axis=3)
    conv7__ = Convolution2D(256,(3, 3), activation='relu', padding='same')(up2__)
    conv7__ = Dropout(0.2)(conv7__)
    conv7__ = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv7__)
    
    up3__ = concatenate([UpSampling2D(size=(2, 2))(conv7__), conv2], axis=3)
    conv8__ = Convolution2D(128,(3, 3), activation='relu', padding='same')(up3__)
    conv8__ = Dropout(0.2)(conv8__)
    conv8__ = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv8__)
    
    up4__ = concatenate([UpSampling2D(size=(2, 2))(conv8__), conv1], axis=3)
    conv9__ = Convolution2D(64,(3, 3), activation='relu', padding='same')(up4__)
    conv9__ = Dropout(0.2)(conv9__)
    conv9__ = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv9__)
    
    out3 = Convolution2D(1, (1,1), activation = 'relu')(conv9__)

    
    model = Model(inputs=inputs, outputs=[out1,out2,out3])
    
    model.compile(loss=['mse','mse','mse'], loss_weights=[0.1,1,1], optimizer= optimizer , metrics=[] )
    
    return model


##############################################################################

def deepDistanceModel(input_height=512, input_width=512, nChannels = 3):

    optimizer = optimizers.Adadelta()
    inputs = Input(shape= (input_height, input_width, nChannels))

    #Encoder (Shared)
    conv1 = Convolution2D(64,(3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128,(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(128,(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(256,(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(256,(3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Convolution2D(512,(3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(512,(3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Convolution2D(1024,(3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(1024,(3, 3), activation='relu', padding='same')(conv5)

    #Decoder 1
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Convolution2D(512,(3, 3), activation='relu', padding='same')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv6)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Convolution2D(256,(3, 3), activation='relu', padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv7)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Convolution2D(128,(3, 3), activation='relu', padding='same')(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv8)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Convolution2D(64,(3, 3), activation='relu', padding='same')(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv9)
    
    out1 = Convolution2D(1, (1,1), activation = 'relu')(conv9)

    #Decoder 2
    up1_ = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6_ = Convolution2D(512,(3, 3), activation='relu', padding='same')(up1_)
    conv6_ = Dropout(0.2)(conv6_)
    conv6_ = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv6_)
    
    up2_ = concatenate([UpSampling2D(size=(2, 2))(conv6_), conv3], axis=3)
    conv7_ = Convolution2D(256,(3, 3), activation='relu', padding='same')(up2_)
    conv7_ = Dropout(0.2)(conv7_)
    conv7_ = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv7_)
    
    up3_ = concatenate([UpSampling2D(size=(2, 2))(conv7_), conv2], axis=3)
    conv8_ = Convolution2D(128,(3, 3), activation='relu', padding='same')(up3_)
    conv8_ = Dropout(0.2)(conv8_)
    conv8_ = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv8_)
    
    up4_ = concatenate([UpSampling2D(size=(2, 2))(conv8_), conv1], axis=3)
    conv9_ = Convolution2D(64,(3, 3), activation='relu', padding='same')(up4_)
    conv9_ = Dropout(0.2)(conv9_)
    conv9_ = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv9_)
    
    out2 = Convolution2D(1, (1,1), activation = 'relu')(conv9_)

    
    model = Model(inputs=inputs, outputs=[out1,out2])

    model.compile(loss=['mse','mse'], optimizer= optimizer , metrics=[] )
    
    return model




