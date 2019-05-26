import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


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


def mask_to_red(mask):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size,img_size)
    c2 = np.zeros((img_size,img_size))
    c3 = np.zeros((img_size,img_size))
    c4 = mask.reshape(img_size,img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


def mask_to_rgba(mask, color='red'):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    zeros = np.zeros((img_size,img_size))
    ones = mask.reshape(img_size,img_size)
    if color == 'red':
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == 'green':
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == 'blue':
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == 'yellow':
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == 'magenta':
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == 'cyan':
        return np.stack((zeros, ones, ones, ones), axis=-1)

    
def plot_imgs(org_imgs, 
              mask_imgs, 
              pred_imgs=None, 
              nm_img_to_plot=10, 
              figsize=4,
              alpha=0.5
             ):
    '''
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    '''
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if  not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

        
    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols*figsize, nm_img_to_plot*figsize))
    axes[0, 0].set_title("original", fontsize=15) 
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15) 
        axes[0, 3].set_title("overlay", fontsize=15) 
    else:
        axes[0, 2].set_title("overlay", fontsize=15) 
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()        
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1

    plt.show()


def zero_pad_mask(mask, desired_size):
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask


def reshape_arr(arr):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def get_cmap(arr):
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'rgb'
        elif arr.shape[3] == 1:
            return 'gray'