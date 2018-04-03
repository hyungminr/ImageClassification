
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from PIL import Image
from scipy.misc import imread, imsave, imresize
from datetime import datetime
from keras.layers import Input, Dense, BatchNormalization, LSTM
from keras.layers import Conv2D, Flatten, Reshape, Add, MaxPooling2D, Dropout
from keras.optimizers import RMSprop
from keras.models import Model
from keras.utils import to_categorical
from models.resnet50 import ResNet50
from models.resnet50_fpn import ResNet50_FPN
from models.vgg16 import VGG16
from models.vgg16_fpn import VGG16_FPN
from models.densenet169 import DenseNet
#get_ipython().magic(u'matplotlib inline')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Training Face Classification Model')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default='0')
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--model', dest='model', help='Network to use [resnet50]',
                        default='resnet50')
    parser.add_argument('--epochs', dest='epochs', help='Number of Epochs for Training',
                        default=500, type=int)
    parser.add_argument('--save', dest='save', help='Save Model File Name',
                        default='result.h5')
    parser.add_argument('--test', dest='test_model', help='Test Model',
                        default=False, type=bool)
    parser.add_argument('--train_data', dest='train_data_dir', help='Training Data Dir',
                        default='kfcf')

    args = parser.parse_args()

    return args


def load_data():

    global img_col, img_row
    
    args = parse_args()
    if args.model == 'densenet':
        img_col = 224
        img_row = 224 # 512 x 512 x 3
    else:
        img_col = 256
        img_row = 256 # 512 x 512 x 3

    data_path = os.path.join(os.getcwd(),'data','wiki') # 데이터 폴더 위치

    text_file = os.path.join(data_path, 'annotation.txt')
    train_data = [i.strip('\n').split('\t') for i in open(text_file)]
    num_data = len(train_data)

    imgs = []
    ages = []
    genders = []
#     num_data = 100
    idxs = range(num_data)
    np.random.seed(0)
    np.random.shuffle(idxs)
    idxs = idxs[:1000]
    idx_count = 0
    for idx in idxs:
        idx_count += 1
        if idx_count % 500 == 0:
            print idx_count
        gender = train_data[idx][0]
        age    = int(float(train_data[idx][1]))
        if gender is '0' or gender is '1':
            gender = int(float(gender))
            if 0 < age and age < 100:
                img_path = os.path.join(os.getcwd(), 'data', 'wiki', train_data[idx][2])
                img = imread(img_path, mode='RGB')
                if img.nbytes > 100000:
                    img = imresize(img, (img_col, img_row, 3))
                    imgs.append(img)
                    ages.append(age)
                    genders.append(gender)

    data = {'Image': imgs,
            'Age': ages,
            'Gender': genders}
    global frame
    frame = pd.DataFrame(data)
    
    return frame#.sample(frac=1).reset_index(drop=True) # 랜덤 셔플링


def show_dataSamples(frame, sample_num=3):
    sample_num = 3
    for idx in xrange(sample_num):
        rand_idx = np.random.randint(len(frame['Movement_id']))
        plt.axis('off')
        plt.title(frame['Movement'][rand_idx])
        _ = plt.imshow(frame['Painting'][rand_idx])
        plt.show()
    return

def construct_data(frame):
    inputData = np.asarray([im for im in frame['Image'].as_matrix()])

    outputData = to_categorical(frame['Gender'])

    for idx in xrange(len(outputData)):
        outputData[idx] = [frame['Gender'][idx] * 100, frame['Age'][idx]]

    input_shape = (img_col, img_row, 3)
    
    return inputData, outputData, input_shape
        
def get_model(input_shape, class_num):
    args = parse_args()
    if args.model == 'resnet50':
        model = ResNet50(include_top=True, weights=None,
                         input_tensor=None, input_shape=input_shape,
                         pooling=None,
                         classes=class_num)
    if args.model == 'resnet50_imagenet':
        model = ResNet50(include_top=False, weights=None,
                         input_tensor=None, input_shape=input_shape,
                         pooling=None,
                         classes=class_num,
                         mode='regression')
        #model.load_weights(os.path.join(os.getcwd(), 'weights', 'imagenet_resnet50.h5'))
        #model.layers.add(Flatten())
        #model.layers.add(Dense(classes, activation='softmax', name='fc_custom'))
                  
    elif args.model == 'vgg16':
        model = VGG16(include_top=True, weights=None,
                      input_tensor=None, input_shape=input_shape,
                      pooling=None,
                      classes=class_num)
    elif args.model == 'vgg16_fpn':
        model = VGG16_FPN(include_top=True, weights=None,
                      input_tensor=None, input_shape=input_shape,
                      pooling=None,
                      classes=class_num)
    elif args.model == 'resnet50_fpn':
        model = VGG16_FPN(include_top=True, weights=None,
                      input_tensor=None, input_shape=input_shape,
                      pooling=None,
                      classes=class_num)
    elif args.model == 'densenet':
        model = DenseNet(nb_dense_block=4, growth_rate=32,
                         nb_filter=64, reduction=0.0,
                         dropout_rate=0.0, weight_decay=1e-4,
                         classes=class_num, weights_path=None)
    rmsprop = RMSprop(lr=1e-05, rho=0.99, epsilon=1e-08, decay=0.001)
    model.compile(optimizer=rmsprop,
                  loss='mean_squared_error',
                  metrics=['mae'])
    print model.summary()
    return model

def train_model(model, batch_size=10):
    args = parse_args()
    epochs = args.epochs
    weightDir = os.path.join(os.getcwd(), 'weights', args.model)
    if not os.path.exists(weightDir):
        os.makedirs(weightDir)
    filepath = weightDir + '/weights.epoch_{epoch:02d}.val-loss_{val_loss:.2f}.h5'
    modelChackpt = keras.callbacks.ModelCheckpoint(filepath,
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=100)
    time_start = datetime.now()
    print '** Training Start, {}'.format(time_start.strftime('%Y-%m-%d %H:%M:%S'))
    history = model.fit(inputData, outputData,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[modelChackpt])
    time_end = datetime.now()
    print '** Training Done, {}'.format(time_end.strftime('%Y-%m-%d %H:%M:%S'))
    time_elapsed = str(time_end - time_start).split(':')
    print '**** Elapsed Time : {} hours {} minutes'.format(time_elapsed[0], time_elapsed[1])
    
    modelName = os.path.join(os.getcwd(), 'weights', args.save)
#    modelName_ = modelName.split('.')
#    modelName_.insert(-1, time_start.strftime('%Y%m%d_%H%M%S'))
#    modelName = '.'.join(modelName_)
    
    print '** Model Saving at [{}] ...'.format(modelName),
    model.save(modelName)
    print 'Done'


def test_model(sample_num=300):
    args = parse_args()
    from keras.models import load_model
    modelName = os.path.join(os.getcwd(), 'weights', args.save)
    model = load_model(modelName)
    
    right = 0
    #f1 = plt.figure(figsize=(20,60))
    for idx in xrange(sample_num):
        rand_idx = np.random.randint(len(frame['Gender']))
        #plt.subplot(10,3,idx+1)
        #plt.axis('off')
        a = frame['Image'][rand_idx]
        a = np.reshape(a, (1, img_col, img_row, 3))
        predict = model.predict(a)
        
        target_age = frame['Age'][rand_idx]
        predicted_age = predict[0][1]
        if np.abs(predicted_age - target_age) < 5:
            right += 1
        #plt.title('     Real Movement : {:<22}\nPredicted Movement : {:<22}'.format(target_mov, predicted_mov))
        #_ = plt.imshow(frame['Painting'][rand_idx])

    print 'Accuracy : {} ({} / {})'.format(right*100./sample_num, right, sample_num)
    #plt.show()
    time_now = datetime.now()
    #f1.savefig('result_' + time_now.strftime('%Y%m%d_%H%M%S') + '.png', dpi=100)


if __name__ == "__main__":
    args = parse_args()
    frame = load_data()
    inputData, outputData, input_shape = construct_data(frame)
    test_model()
