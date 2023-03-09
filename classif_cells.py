


# from google.colab import drive
# drive.mount('/content/drive')

## Téléchargement de la base de données
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
#!git clone https://github.com/fabiopereira59/abeilles-cap500
IMG_SIZE = 224
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='/home/lucien/Documents/cells_malaria/fish_data_split/train',
    labels='inferred',
    label_mode='categorical',
    shuffle = False,
    batch_size=16,
    image_size=(IMG_SIZE, IMG_SIZE))
class_names = train_ds.class_names
print(class_names)
nb_classes = len(class_names)
print(nb_classes)
## Chargement des données
from tensorflow import keras
from tensorflow.keras import layers
# Paramètres
IMG_SIZE = 224 # pour utiliser ResNet
# Récupération des dataset pour l'entraînement (train, val)
# Shuffle à false pour avoir accès aux images depuis
# leur chemin d'accès avec train_ds.file_paths
train_ds = keras.utils.image_dataset_from_directory(
    directory='/home/lucien/Documents/cells_malaria/fish_data_split/train',
    labels='inferred',
    label_mode='categorical',
    shuffle = False,
    batch_size=16,
    image_size=(IMG_SIZE, IMG_SIZE))

validation_ds = keras.utils.image_dataset_from_directory(
    directory='/home/lucien/Documents/cells_malaria/fish_data_split/val',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(IMG_SIZE, IMG_SIZE))
## Augmentation de données : Sequence et Albumentations
# !pip uninstall opencv-python-headless==4.5.5.62
# !pip install opencv-python-headless==4.1.2.30
# !pip install -q -U albumentations
# !echo "$(pip freeze | grep albumentations) is successfully installed"
from albumentations import (Compose, Rotate, HorizontalFlip, VerticalFlip, Affine, RandomBrightnessContrast, ChannelShuffle)
import albumentations as A

AUGMENTATIONS_TRAIN = Compose([
    Rotate(limit=[0,100], p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Affine(shear=[-45, 45], p=0.5),
    RandomBrightnessContrast(p=0.5)
])
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2 as cv

class AbeillesSequence(Sequence):
    # Initialisation de la séquence avec différents paramètres
    def __init__(self, x_train, y_train, batch_size, augmentations):
        self.x_train = x_train
        self.y_train = y_train
        self.classes = class_names
        self.batch_size = batch_size
        self.augment = augmentations
        self.indices1 = np.arange(len(x_train))
        np.random.shuffle(self.indices1) # Les indices permettent d'accéder
        # aux données et sont randomisés à chaque epoch pour varier la composition
        # des batches au cours de l'entraînement

    # Fonction calculant le nombre de pas de descente du gradient par epoch
    def __len__(self):
        return int(np.ceil(x_train.shape[0] / float(self.batch_size)))
    
    # Application de l'augmentation de données à chaque image du batch
    def apply_augmentation(self, bx, by):

        batch_x = np.zeros((bx.shape[0], IMG_SIZE, IMG_SIZE, 3))
        batch_y = by
        
        # Pour chaque image du batch
        for i in range(len(bx)):
            class_labels = []
            class_id = np.argmax(by[i])
            class_labels.append(self.classes[class_id])

            # Application de l'augmentation à l'image
            img = cv.imread(bx[i])
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            transformed = self.augment(image=img)
            
            #on veut que les images soient de taille 224x224
            img = cv.resize(transformed['image'], (IMG_SIZE, IMG_SIZE))
            batch_x[i] = img
              
        return batch_x, batch_y

    # Fonction appelée à chaque nouveau batch : sélection et augmentation des données
    # idx = position du batch (idx = 5 => on prend le 5ème batch)
    def __getitem__(self, idx):
        batch_x = self.x_train[self.indices1[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_y = self.y_train[self.indices1[idx * self.batch_size:(idx + 1) * self.batch_size]]
           
        batch_x, batch_y = self.apply_augmentation(batch_x, batch_y)

        # Normalisation des données
        batch_x = tf.keras.applications.resnet.preprocess_input(batch_x)
        
        return batch_x, batch_y

    # Fonction appelée à la fin d'un epoch ; on randomise les indices d'accès aux données
    def on_epoch_end(self):
        np.random.shuffle(self.indices1)
# Les images sont stockées avec les chemins d'accès
import numpy as np
print(nb_classes)

x_train = np.array(train_ds.file_paths)
y_train = np.zeros((258, nb_classes))#rentrer la taille du train

ind_data = 0
for bx, by in train_ds.as_numpy_iterator():
  y_train[ind_data:ind_data+bx.shape[0]] = by
  ind_data += bx.shape[0]
# Instanciation de la Sequence
train_ds_aug = AbeillesSequence(x_train, y_train, batch_size=16, augmentations=AUGMENTATIONS_TRAIN)

# Normalisation des données de validation
import numpy as np
import tensorflow as tf

n = 32
x_val = np.zeros((n, IMG_SIZE, IMG_SIZE, 3))#rentrer la taille du val
y_val = np.zeros((n, nb_classes))#rentrer la taille du val

ind_data = 0
for bx, by in validation_ds.as_numpy_iterator():
  x_val[ind_data:ind_data+bx.shape[0]] = bx
  y_val[ind_data:ind_data+bx.shape[0]] = by
  ind_data += bx.shape[0]

x_val = tf.keras.applications.resnet.preprocess_input(x_val)
## Création du modèle
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow as tf
### Poids d'imagenet
conv_base = keras.applications.resnet.ResNet101(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
    classes=nb_classes,
)

model = keras.Sequential(
    [
        conv_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(nb_classes, kernel_regularizer=regularizers.L2(1e-4), activation='softmax')
    ]
)
model.summary()

import pandas as pd
import numpy as np

from numpy.ma.core import transpose
from keras import backend as K
import math
import tensorflow as tf

## Entraînement du modèle
# Ajout de l'optimiseur, de la fonction coût et des métriques
lr = 1e-3
model.compile(optimizers.SGD(learning_rate=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Les callbacks, là où on sauvegarde les poids du réseau

#filepath = path to save the model at the end of each epoch

model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/lucien/Documents/cells_malaria/model_fish/model', 
    save_weights_only=True,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

#early_stopping_cb = tf.keras.callbacks.EarlyStopping(
#    monitor="val_categorical_accuracy",
#    min_delta=0.01,
#    patience=8,
#    verbose=1,
#    mode="auto")
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1,
                              patience=5, min_lr=0.00001, verbose=1)
history = model.fit(train_ds_aug, epochs=10, validation_data = (x_val, y_val), callbacks=[model_checkpoint_cb, reduce_lr_cb]) 

