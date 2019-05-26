import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def augment_data(
    X_train,
    Y_train,
    batch_size=32,
    seed=0,
    data_gen_args = dict(
        rotation_range = 10.,
        height_shift_range = 0.02,
        shear_range = 5,
        horizontal_flip = True,
        vertical_flip = False,
        fill_mode = 'constant'
    )):
    
    
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size,
shuffle=True, seed=seed)
    
    train_generator = zip(X_train_augmented, Y_train_augmented)
    
    return train_generator