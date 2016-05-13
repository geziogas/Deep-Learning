from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.visualize_util import plot # To plot
from keras.models import model_from_json
# import pandas as pd

train_data = np.load('HWDB1.1trn_gnt/30-class-trainset-shuffled.npy')
test_data = np.load('HWDB1.1tst_gnt/30-class-testset-shuffled.npy')

train_labels = np.load('HWDB1.1trn_gnt/30-class-trainlabels-shuffled.npy')
test_labels = np.load('HWDB1.1tst_gnt/30-class-testlabels-shuffled.npy')

ind,x,y = train_data.shape
it,xt,yt = test_data.shape

# input image dimensions
img_rows, img_cols = x, y
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

batch_size = 50
nb_epoch = 100
nb_classes=30 # OR len(set(train_labels))

train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255.0
test_data /= 255.0
print('X_train shape:', train_data.shape)
print(train_data.shape[0], 'train samples')
print(test_data.shape[0], 'test samples')

# Factorize labels to number classes
# labToNum_trn,l_unique_trn = pd.factorize(train_labels)
# labToNum_tst,l_unique_tst = pd.factorize(test_labels)

# # Correction of indices mapping for training set
# rInd2=[]
# for i in range (len(l_unique_trn)):
#     for j in range(len(l_unique_tst)):
#         if(l_unique_tst[i]==l_unique_trn[j]):
#             rInd2.append(j)

# newY_test = []
# for i in range(len(labToNum_tst)):
#     newY_test.append(rInd2[labToNum_tst[i]])

# reY_test = np.asarray(newY_test)
# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(labToNum_trn, nb_classes)
# Y_test = np_utils.to_categorical(reY_test, nb_classes)

# np.save('Y_train.npy',Y_train)
# np.save('Y_test.npy',Y_test)


# Load the pre-processed data-sets ready to be used
Y_train = np.load('Y_train.npy')
Y_test = np.load('Y_test.npy')




#CNN Model
# Model: 11 Weighted Layers
model = Sequential()

# CNN-1: 3-64x2
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

# CNN-2: 3-128x2
model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

# CNN-3: 3-256x2
model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*4, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

# # CNN-4: 3-512x2
model.add(Convolution2D(nb_filters*8, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*8, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

# FC-1024x2 Fully connected layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# FC-30 Last layer
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# Load Model...
model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

# Adadelta optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

# history = model.fit(train_data, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_split=0.2)

history = model.fit(train_data, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_split=0.2)

from keras.models import model_from_json
json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5',overwrite=True)

score = model.evaluate(test_data, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Example
s = model.predict(test_data[0:20]) #predict
res = np.argmax(s,axis=1) #return the indices of labels
# Use slab_tst[indices] to validate


# for i in range(res.shape[0]):
#     print(i,test_labels[i],'Class:',reY_test[i],\
#     ',Predicted:',l_unique_trn[res[i]],'Class:',\
#     res[i],',Result:',reY_test[i]==res[i])



# np.where(labToNum_trn==23)
# np.where(reY_test==23)
# # 1-by-1 image check
# plt.subplot(1,3,1)
# plt.imshow(test_data[3,0,:,:],cmap='Greys_r',title='A')
# plt.subplot(1,3,2)
# plt.imshow(train_data[39,0,:,:],cmap='Greys_r')
# plt.subplot(1,3,3)
# plt.imshow(test_data[18,0,:,:],cmap='Greys_r')
# plt.show()