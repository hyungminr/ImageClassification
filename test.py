# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from scipy.misc import imread, imsave, imresize
from keras.optimizers import RMSprop
from keras.models import Model, load_model
from keras.utils import to_categorical
from models.resnet50 import ResNet50

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Training Image Classification Model')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default='0')
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--saved', dest='saved', help='Save Model File Name',
                        default='resnet50.imagenet.h5')
    parser.add_argument('--data', dest='data', help='Test Data Dir',
                        default='example_annotation')
    parser.add_argument('--size', dest='img_size', help='Size of Input Images (default:224 -> 224x224x3)',
                        default=224, type=int)

    args = parser.parse_args()

    return args

def load_data():
    data_path = os.path.join(os.getcwd(),'data',args.data)
    text_file = os.path.join(data_path, 'annotation.txt')
    if os.path.exists(text_file):
        return load_data_from_annotation()
    else:
        return load_data_from_categorical_folders()

def load_data_from_annotation():
    
    args = parse_args()
    img_col = args.img_size
    img_row = args.img_size

    data_path = os.path.join(os.getcwd(),'data',args.data)

    text_file = os.path.join(data_path, 'annotation.txt')
    train_data = [i.strip('\n').split('\t') for i in open(text_file)]
    num_data = len(train_data)

    imgs = []
    labels = []
    
    
    idxs = range(num_data)
    np.random.seed(0)
    np.random.shuffle(idxs)
    idx_count = 0
    for idx in tqdm(idxs):
        img_path = train_data[idx][0]
        label    = train_data[idx][1][:-1] # remove \r
        img_path = os.path.join(os.getcwd(), 'data', args.data, img_path)
        img = imread(img_path, mode='RGB')
        if img.nbytes > 10**3: # Read image file only when its size is larger than 1 kilo bytes
            img = imresize(img, (img_col, img_row, 3))
            imgs.append(img)
            labels.append(label)
        else:
            continue

    data = {'Image': imgs,
            'Label': labels}
    
    return pd.DataFrame(data)
    
def load_data_from_categorical_folders():

    args = parse_args()
    img_col = args.img_size
    img_row = args.img_size
    
    img_files = []
    labels = []
    
    data_path = os.path.join(os.getcwd(), 'data', args.data)
    category_idx = 0

    print 'Loading files from data folder'
    for category in tqdm(os.listdir(data_path)): # categorical subfolders in data folder
        img_path = os.path.join(data_path, category)
        img_count = 0
        for img_file in os.listdir(img_path):
            is_image = (img_file.endswith('png') or
                        img_file.endswith('PNG') or
                        img_file.endswith('jpg') or
                        img_file.endswith('JPG') or
                        img_file.endswith('jpeg') or
                        img_file.endswith('JPEG') or
                        img_file.endswith('gif') or
                        img_file.endswith('GIF') or
                        img_file.endswith('tif') or
                        img_file.endswith('TIF'))
                        
            if is_image:
                img_count += 1
                img_files.append(os.path.join(img_path, img_file))
                labels.append(category)

        subcategory = [os.path.join(img_path, subfolder) for subfolder in os.listdir(img_path) if os.path.isdir(os.path.join(img_path,subfolder))]
        for subcat in subcategory:
            sub_path = os.path.join(img_path, subcat)
            for img_file in os.listdir(sub_path):
                is_image = (img_file.endswith('png') or
                            img_file.endswith('PNG') or
                            img_file.endswith('jpg') or
                            img_file.endswith('JPG') or
                            img_file.endswith('jpeg') or
                            img_file.endswith('JPEG') or
                            img_file.endswith('gif') or
                            img_file.endswith('GIF') or
                            img_file.endswith('tif') or
                            img_file.endswith('TIF'))
                if is_image:
                    img_count += 1
                    img_files.append(os.path.join(img_path, img_file))
                    labels.append(category)
    imgs = []
    new_labels = []
    num_data = len(img_files)
    idxs = range(num_data)
    np.random.seed(0)
    np.random.shuffle(idxs)
    print 'Image reading from files'
    for idx in tqdm(idxs):
        img_path = img_files[idx]
        label    = labels[idx]
        img = imread(img_path, mode='RGB')
        if img.nbytes > 10**3: # Read image file only when its size is larger than 1 kilo bytes
            img = imresize(img, (img_col, img_row, 3))
            imgs.append(img)
            new_labels.append(label)
        else:
            continue

    data = {'Image': imgs,
            'Label': new_labels}
    return pd.DataFrame(data)


def label_to_categorical(labels, label_dict=None, mode='to_categorical'):
    if mode == 'to_categorical':
        label_dict, ids = np.unique(labels, return_inverse=True)
        output = to_categorical(ids, len(label_dict))
    else:
        output = label_dict[labels.argmax(1)]
        output = output[0]
    return output, label_dict


def construct_data(frame):
    args    = parse_args()
    img_col = args.img_size
    img_row = args.img_size
    inputData   = np.asarray([im for im in frame['Image'].as_matrix()])
    outputData, label_dict  = label_to_categorical(frame['Label'])
    input_shape = (img_col, img_row, 3)
    return inputData, outputData, input_shape, label_dict
        
    
def test_model(label_dict, sample_num=4):
    print '**** Start testing the model'
    args = parse_args()
    img_col = args.img_size
    img_row = args.img_size
    
    modelName = os.path.join(os.getcwd(), 'weights', args.saved)
    model = load_model(modelName)
    
    right = 0
    f1 = plt.figure(figsize=(20,20))
    
    num_data = len(frame['Label'])
    
    idxs = range(int(num_data*0.8), num_data)
    np.random.seed(0)
    np.random.shuffle(idxs)
    idx_count = 0
    for idx in tqdm(idxs):
        plt.subplot(int(np.ceil(sample_num/2.)),2,idx_count+1)
        plt.axis('off')
        img = frame['Image'][idx]
        input_img = np.reshape(img, (1, img_col, img_row, 3))
        predict = model.predict(input_img)
        
        target_label = frame['Label'][idx]
        predicted_label = label_to_categorical(predict,label_dict,'reverse')
        predicted_label = predicted_label[0]
        if predicted_label == target_label:
            right += 1
        plt.title('Groundtruth : {}\nPredicted   : {}'.format(target_label, predicted_label))
        _ = plt.imshow(img)
        idx_count += 1
        if idx_count >= sample_num:
            break

    print '** Accuracy : {} ({} / {})'.format(right*100./sample_num, right, idx_count)
    now = datetime.now()
    nowDatetime = now.strftime('%Y_%m_%d-%H%M%S')
    imgResultFileName = os.path.join(os.getcwd(), 'result', nowDatetime + '_' + args.saved + '.png')
    f1.savefig(imgResultFileName, dpi=100)
    print '**** Result Image Saved at [{}] ...'.format(imgResultFileName)


if __name__ == "__main__":
    args = parse_args()
    frame = load_data()
    inputData, outputData, input_shape, label_dict = construct_data(frame)
    test_model(label_dict)