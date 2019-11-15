import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate


def upsample(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)
    #return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def conv2d_block(
    inputs, 
    use_batch_norm=False, 
    dropout=0.0, 
    filters=[], 
    kernel_size=(3,3), 
    activation='relu', 
    kernel_initializer='he_normal', 
    padding='same'):
    
    c = inputs
    for i in range(len(filters)):
        c = Conv2D(filters[i], kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (c)
        if use_batch_norm:
            c = BatchNormalization()(c)
        if dropout > 0.0 and i != len(filters) - 1:
            c = Dropout(dropout)(c)

    return c

def create_unet(
    input_shape,
    filters=[[48, 48], [48], [48], [48], [48], [48], [96, 96], [96, 96], [96, 96], [96, 96], [64, 32], [3]],
    skip_start=True,
    use_batch_norm=False, 
    use_dropout_on_upsampling=False, 
    dropout=0.0, 
    dropout_change_per_layer=0.0):

    num_layers = int(len(filters) / 2 - 1)

    inputs = Input(input_shape)
    x = inputs   

    skip_cons = []
    if (skip_start):
        skip_cons.append(x)

    filter_index = 0
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters[filter_index], use_batch_norm=use_batch_norm, dropout=dropout)

        if (not(skip_start)):
            skip_cons.append(x)
        
        x = MaxPooling2D((2, 2)) (x)

        if (skip_start and l != num_layers - 1):
            skip_cons.append(x)

        dropout += dropout_change_per_layer

        filter_index += 1

    x = conv2d_block(inputs=x, filters=filters[filter_index], use_batch_norm=use_batch_norm, dropout=dropout)
    filter_index += 1
    
    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(skip_cons):        
        dropout -= dropout_change_per_layer

        x = upsample(None, (2, 2), strides=(2, 2), padding='same') (x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters[filter_index], use_batch_norm=use_batch_norm, dropout=dropout)

        filter_index += 1

    output_dim = filters[filter_index][0]
    outputs = Conv2D(output_dim, (1, 1)) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model