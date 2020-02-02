"""
    Code taken from :
    https://colab.research.google.com/drive/1wCKUsNJYoXxJTZXu-B_qzT6huDeSXLlp?ts=5e1a82f1
    Work done by Abdiel 
"""

from tensorflow import keras as K

def unetl2(input_size=(256, 256, 3)):
  # Take in the inputs
    inputs = K.layers.Input(input_size)
    # First encoder block
    conv_1 = K.layers.Conv2D(64, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(inputs)
    conv_1 = K.layers.Conv2D(64, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_1)
    pool_1 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_1)
    # Second encoder block
    conv_2 = K.layers.Conv2D(128, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(pool_1)
    conv_2 = K.layers.Conv2D(128, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_2)
    pool_2 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)
    # Third encoder block
    conv_3 = K.layers.Conv2D(256, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(pool_2)
    conv_3 = K.layers.Conv2D(256, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_3)
    pool_3 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_3)
    # Fourth encoder block
    conv_4 = K.layers.Conv2D(512, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(pool_3)
    conv_4 = K.layers.Conv2D(512, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_4)
#drop_4 = K.layers.Dropout(0.5)(conv_4)
#pool_4 = K.layers.MaxPooling2D(pool_size=(2, 2))(drop_4)
    pool_4 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_4)
    # Encoder-decoder conection
    conv_5 = K.layers.Conv2D(1024, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(pool_4)
    conv_5 = K.layers.Conv2D(1024, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_5)
#drop_5 = K.layers.Dropout(0.5)(conv_5)
    # First decoder block
#up_6 = K.layers.UpSampling2D(size=(2, 2))(drop_5)
    up_6 = K.layers.UpSampling2D(size=(2, 2))(conv_5)
    up_6 = K.layers.Conv2D(512, 2, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(up_6)
    # Concatenation of first decoder and fourth encoder blocks
    merge_6 = K.layers.Concatenate()([conv_4, up_6])
    conv_6 = K.layers.Conv2D(512, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(merge_6)
    conv_6 = K.layers.Conv2D(512, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_6)
    # Second decoder block
    up_7 = K.layers.UpSampling2D(size=(2, 2))(conv_6)
    up_7 = K.layers.Conv2D(256, 2, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(up_7)
    # Concatenation of second decoder and third encoder block
    merge_7 = K.layers.Concatenate()([conv_3, up_7])
    conv_7 = K.layers.Conv2D(256, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(merge_7)
    conv_7 = K.layers.Conv2D(256, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_7)
    # Third decoder block
    up_8 = K.layers.UpSampling2D(size=(2, 2))(conv_7)
    up_8 = K.layers.Conv2D(128, 2, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(up_8)
    # Concatenation of third decoder and second encoder block
    merge_8 = K.layers.Concatenate()([conv_2, up_8])
    conv_8 = K.layers.Conv2D(128, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(merge_8)
    conv_8 = K.layers.Conv2D(128, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_8)
    # Fourth decoder block
    up_9 = K.layers.UpSampling2D(size=(2, 2))(conv_8)
    up_9 = K.layers.Conv2D(64, 2, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(up_9)
    # Concatenation of fourth decoder and first encoder block
    merge_9 = K.layers.Concatenate()([conv_1, up_9])
    conv_9 = K.layers.Conv2D(64, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(merge_9)
    conv_9 = K.layers.Conv2D(64, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_9)
    # Output of U-Net
    conv_9 = K.layers.Conv2D(2, 3, activation="relu", padding="same",kernel_regularizer=K.regularizers.l2(0.005))(conv_9)
    conv_10 = K.layers.Conv2D(1, 1, activation="sigmoid")(conv_9)

    model = K.models.Model(inputs=[inputs], outputs=[conv_10])
    return model

