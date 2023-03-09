# Téléchargement de la base de données
# Pour récupérer le nombre de classes du training dataset
from tensorflow import keras
from tensorflow.keras import layers
#!git clone https://github.com/fabiopereira59/abeilles-cap500
IMG_SIZE = 224
train_ds = keras.utils.image_dataset_from_directory(
    directory='/home/lucien/Documents/cells_malaria/fish_data_split/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(IMG_SIZE, IMG_SIZE))
class_names = train_ds.class_names
print(class_names)
nb_classes = len(class_names)
print(nb_classes)
# Chargement du modèle
# from google.colab import drive
# drive.mount('/content/drive')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow as tf
# Création de l'architecture du modèle à utiser

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

# Chargement des poids
model.load_weights('/home/lucien/Documents/cells_malaria/model_fish/model')#mettre ici le même chemin que celui des callbacks (dans la var model_checkpoints)


# Prédiction sur une image

import numpy as np

saumon = '/home/lucien/Documents/cells_malaria/fish-0c33b5838238321edd4ec901d7a1c278c984e232fa2de3f4e4212b06345f05bf.jpg'
anguille = '/home/lucien/Documents/cells_malaria/fish-0b6a181f476f7d683af9944cf948455a40626bf54385f5a427afa056468779c1.jpg'

from tensorflow.keras.preprocessing import image

img = image.load_img(saumon, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

print(  "This image most likely belongs to {} with a {:.2f} percent confidence."    
        .format(class_names[np.argmax(score)], 100 * np.max(score)))

# maintenant on veut afficher l'image avec la prédiction infected or not infected 

import matplotlib.pyplot as plt

#on veut afficher l'image en écrivant en légende "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)

plt.imshow(img)
plt.title("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
plt.show()

