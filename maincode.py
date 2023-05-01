import tensorflow as tf
from tensorflow import keras
import keras_preprocessing
from tensorflow.keras.models import Model
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
from tqdm import tqdm 
import cv2 
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from os import listdir


# After loading the modules why need to specify the training and  testing data
TRAINING_DIR = ""
training_datagen = ImageDataGenerator(rescale=1./255,
zoom_range=0.15,
horizontal_flip=True,
fill_mode='nearest')

VALIDATION_DIR = ""
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
TRAINING_DIR,
#target_size=(256,256),
shuffle = True,
class_mode='categorical',
batch_size = 20)

validation_generator = validation_datagen.flow_from_directory(
VALIDATION_DIR,
#target_size=(256,256),  
class_mode='categorical',
shuffle = True,
batch_size= 15)


#After that we run the inception V3 model

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, input_shape=(5,),activation='relu')(x)
x = Dropout(0.2)(x)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history= model.fit(
train_generator,
steps_per_epoch = 4,
epochs = 20,
validation_data = validation_generator,
validation_steps = 4)

# we check with a graph about how well our training and validation accurancy went

acc = history.history['acc']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
loss = history.history['loss']
epochs = range(len(acc))
plt.figure(figsize=[7,5])
plt.plot(epochs, acc, 'r', label='Training_accurancy')
plt.plot(epochs, val_acc, 'b', label='validation_accurancy')
plt.title("Validation and training accurancy")
plt.legend()
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
#plt.legend(loc=0)
plt.figure()
plt.show()

#We freeze the model and we run it again in order to enhance the validation and training accurancy
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(
train_generator,
steps_per_epoch = 4,
epochs = 20,
validation_data = validation_generator,
validation_steps = 4)

# we save our model.
model.save('inceptionv3.h5')


#Finally we define our predict funvtion which it will take a given image and predicts if there is a fire in the image or not. It also maps the fire region based on the pixel values.

def predict(img_rel_path):
    img = image.load_img(img_rel_path)
    read_image = cv2.imread(img_rel_path)
    read_image1 = cv2.imread(img_rel_path,0)
    img = image.img_to_array(img, dtype=np.uint)
 
   
    img = np.array(img)/255.0
    width, height, c = img.shape
    p = model.predict(img[np.newaxis, ...])
    labels = {0: 'fire', 1: 'Non fire'}
    patch_size = 256
    n_classes = 2

    predicted_class = labels[np.argmax(p[0], axis=-1)]
    if predicted_class == 'fire':
        print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
        print("Classified:", predicted_class, "\n\n")
        _, thresholded_image = cv2.threshold(read_image1, 220, 255, cv2.THRESH_BINARY)
       
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,3.5))
        
        ax[0].imshow(img.squeeze())
        ax[0].set_title('Original image',color='C0')
        ax[0].axis('off')
        
        ax[1].imshow(thresholded_image,cmap='rainbow')
        ax[1].set_title('Predicted image',color='C0')
        red_patch = mpatches.Patch(color='red', label='Fire region')
        plt.legend(handles=[red_patch], loc='center left', bbox_to_anchor=(1, 0.5))
       
        ax[1].axis('off')
        plt.subplots_adjust()
        plt.show()
    else:
        print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
        print("Classified:", predicted_class, "\n\n")
        plt.title("No fire in the area")
        plt.axis('off')
        plt.imshow(img.squeeze())
        plt.show()
    def get_prediction_class(prediction):
        if prediction[0] > 0.5:
            return 'Class 1'
        elif prediction[1] > 0.5:
            return 'Class 2'
        

    classes=[]
    prob=[]
    print("\n-------------------Individual Probability--------------------------------\n")

    for i,j in enumerate (p[0],0):
        print(labels[i].upper(),':',round(j*100,2),'%')
        classes.append(labels[i])
        prob.append(round(j*100,2))



acc = history.history['acc']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
loss = history.history['loss']
epochs = range(len(loss))
plt.figure(figsize=[6,5])
plt.plot(epochs, acc, 'r', label='Training_accurancy')
plt.plot(epochs, val_acc, 'b', label='validation_accurancy')
plt.title("Validation and training accurancy")
plt.legend()
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
#plt.legend(loc=0)
plt.figure()
plt.show()


predict()
