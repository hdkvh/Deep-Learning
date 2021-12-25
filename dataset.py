import cv2
import numpy as np
from scipy import io
import os, shutil


def load_hoda(training_sample_size=1000, test_sample_size=200, size=5):
    #load dataset
    trs = training_sample_size
    tes = test_sample_size
    dataset = io.loadmat('D://University//PHD//Simulations//Temp and test//dataset/Data_hoda_full.mat')

    #test and training set
    X_train_orginal = np.squeeze(dataset['Data'][:trs])
    y_train = np.squeeze(dataset['labels'][:trs])
    X_test_original = np.squeeze(dataset['Data'][trs:trs+tes])
    y_test = np.squeeze(dataset['labels'][trs:trs+tes])

    #resize
    X_train_5by5 = [cv2.resize(img, dsize=(size, size)) for img in X_train_orginal]
    X_test_5by5 = [cv2.resize(img, dsize=(size, size)) for img in X_test_original]
    #reshape
    X_train = [x.reshape(size*size) for x in X_train_5by5]
    X_test = [x.reshape(size*size) for x in X_test_5by5]
    
    return X_train, y_train, X_test, y_test


def load_catVSdog (trsize = 1000, vlsize = 500, tesize = 500):
    # The path to the directory where the original
    # dataset was uncompressed
    original_dataset_dir = 'D:\\University\\PHD\\Datasets\\Kaggle\\dogs-vs-cats\\train'
    
    # The directory where we will
    # store our smaller dataset
    base_dir = 'D:\\University\\PHD\\Datasets\\Kaggle\\dogs-vs-cats\\catVsdog'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    # Directories for our training,
    # validation and test splits
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    
    validation_dir = os.path.join(base_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
        
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        
    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)

    
    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)

    
    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)

    
    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)
    
    # Directory with our validation cat pictures
    test_cats_dir = os.path.join(test_dir, 'cats')
    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)
    
    # Directory with our validation dog pictures
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)
    
    # Copy first 1000 cat images to train_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(trsize)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    # Copy next 500 cat images to validation_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(trsize, trsize + vlsize)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
        
    # Copy next 500 cat images to test_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(trsize + vlsize, trsize + vlsize + tesize)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
        
    # Copy first 1000 dog images to train_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(trsize)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
        
    # Copy next 500 dog images to validation_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(trsize, trsize + vlsize)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
        
    # Copy next 500 dog images to test_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(trsize + vlsize,  trsize + vlsize + tesize)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
    
    
    train_cats_dir
    
    print('total training cat images:', len(os.listdir(train_cats_dir)))
    
    print('total training dog images:', len(os.listdir(train_dogs_dir)))
    
    print('total validation cat images:', len(os.listdir(validation_cats_dir)))
    
    print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
    
    print('total test cat images:', len(os.listdir(test_cats_dir)))
    
    print('total test dog images:', len(os.listdir(test_dogs_dir)))
    
    return train_dir, validation_dir, test_dir