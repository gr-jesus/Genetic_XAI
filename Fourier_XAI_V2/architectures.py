# In this part of the library you can use some popular architectures
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers

def VGG16(n_classes):
    model=Sequential()
    # Block 1
    model.add(Conv2D(input_shape=(256,256,3), filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_Conv1'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block1_pool'))
    # Block_2
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_Conv1'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block2_pool'))
    # Block_3
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_Conv1'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_Conv2'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_Conv3'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block3_pool'))
    # Block_4 
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='block4_Conv1'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='block4_Conv2'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='block4_Conv3'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block4_pool'))
    # Block_5
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_Conv1'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_Conv2'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_Conv3'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block5_pool'))
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu', name='fc1'))
    model.add(Dense(units=4096, activation='relu', name='fc2'))
    model.add(Dense(units=n_classes, activation='softmax', name='output'))
    return model

def VGG13(n_classes):
    model=Sequential()
    # Block 1
    model.add(Conv2D(input_shape=(256,256,3), filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_Conv1'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block1_pool'))
    # Block_2
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_Conv1'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block2_pool'))
    # Block_3
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_Conv1'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block3_pool'))
    # Block_4 
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='block4_Conv1'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name='block4_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block4_pool'))
    # Block_5
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_Conv1'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block5_pool'))
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu', name='fc1'))
    model.add(Dense(units=4096, activation='relu', name='fc2'))
    model.add(Dense(units=n_classes, activation='softmax', name='output'))
    return model


def VGG11(n_classes):
    model=Sequential()
    # Block 1
    model.add(Conv2D(input_shape=(256,256,1), filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block_1_Conv1'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block1_pool'))
    # Block 2
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_Conv1'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block2_pool'))
    # Block 3
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_Conv1'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block3_pool'))
    # Block_4 
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_Conv1'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block4_pool'))
    # Block_5
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_Conv1'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_Conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block5_pool'))
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu', name='fc1'))
    model.add(Dense(units=4096, activation='relu', name='fc2'))
    model.add(Dense(units=n_classes, activation='softmax', name='output', kernel_regularizer=regularizers.L2()))
    return model

def alexNet(n_classes):
    model=Sequential()
    #First Layer
    model.add(Conv2D(input_shape=(256,256,1), strides=(4,4), filters=96, kernel_size=(3,3), padding='valid', activation='relu', name='Conv1'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu', name='Conv2'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='Conv3'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='Conv4'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='Conv5'))

    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu', name='fc1'))
    model.add(Dropout(0.4))
    model.add(Dense(units=1024, activation='relu', name='fc2'))
    model.add(Dropout(0.4))
    model.add(Dense(units=n_classes, activation='softmax', name='output', kernel_regularizer=regularizers.L2()))
    return model

