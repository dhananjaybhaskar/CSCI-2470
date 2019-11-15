import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import tensorflow as tf
import numpy as np

from unet_model import create_unet

def main():
    '''
    if len(sys.argv) != 2 or sys.argv[1] not in {"rain", "snow"}:
        print("USAGE: python train.py <Model Type>")
        print("<Model Type>: [rain/snow]")
        exit()
        '''

    model = create_unet(input_shape=(256, 256, 3),
                        filters=[[48, 48], [48], [48], [48], [48], [48], [96, 96], [96, 96], [96, 96], [96, 96], [64, 32], [3]],
                        skip_start=True,
                        use_batch_norm=False, 
                        use_dropout_on_upsampling=False, 
                        dropout=0.0, 
                        dropout_change_per_layer=0.0)

    model.summary()

if __name__ == '__main__':
    main()