# Image Classification on Rice Leaf Diseases
- Dataset acquaired from my google.drive https://drive.google.com/file/d/1guJGuOTA4oHIiJeyaucPe3cgTwi-c3LT/view?usp=sharing

## Objective
Classify three diseases on rice leaf there are :
- Bacterial leaf blight 

  [![Bacteria-leaf-blight-4.jpg](https://i.postimg.cc/Y9TWtTX6/Bacteria-leaf-blight-4.jpg)](https://postimg.cc/34CRBfQN)

- Brown Spot

  [![brown-spot-45.jpg](https://i.postimg.cc/mkGgX4X5/brown-spot-45.jpg)](https://postimg.cc/S2D4RBp6)

- Leaf smut

  [![leaf-smut-61-ext.jpg](https://i.postimg.cc/fyZM7Kk4/leaf-smut-61-ext.jpg)](https://postimg.cc/R6sr4wcG)

## Data Augmentation
i do explicitly augmentation data using PIL library to do it.
- cropping the images
- change the brightness
- change the saturation
- rotating the images

## Data Preparation
Load the images
```
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random

brown_spot_dir = Path("../tmp/rice_leaf_diseases_8/train/Brown spot/")
brown_spot = list(brown_spot_dir.glob(r"**/*.jpg")) + list(brown_spot_dir.glob(r"**/*.JPG"))
print("brown spot : ",len(brown_spot))

bacterial_leaf_blight_dir = Path("../tmp/rice_leaf_diseases_8/train/Bacterial leaf blight/")
bacterial_leaf_blight = list(bacterial_leaf_blight_dir.glob(r"**/*.jpg")) + list(bacterial_leaf_blight_dir.glob(r"**/*.JPG"))
print("bacterial leaf blight : ",len(bacterial_leaf_blight))

leaf_smut_dir = Path("../tmp/rice_leaf_diseases_8/train/Leaf smut/")
leaf_smut = list(leaf_smut_dir.glob(r"**/*.jpg")) + list(leaf_smut_dir.glob(r"**/*.JPG"))
print("leaf smut : ", len(leaf_smut))

healthy_dir = Path("../tmp/rice_leaf_diseases_8/train/Healthy/")
healthy = list(healthy_dir.glob(r"**/*.jpg")) + list(healthy_dir.glob(r"**/*.JPG"))
print("healthy : ", len(healthy))

train_dataset = brown_spot + bacterial_leaf_blight + leaf_smut + healthy
print(len(train_dataset))
```
```
brown spot :  3510
bacterial leaf blight :  3510
leaf smut :  3510
healthy :  3509
14039
```
```
test_brown_spot_dir = Path("../tmp/rice_leaf_diseases_8/test/Brown spot/")
test_brown_spot = list(test_brown_spot_dir.glob(r"**/*.jpg")) + list(test_brown_spot_dir.glob(r"**/*.JPG"))
print("brown spot : ",len(test_brown_spot))

test_bacterial_leaf_blight_dir = Path("../tmp/rice_leaf_diseases_8/test/Bacterial leaf blight/")
test_bacterial_leaf_blight = list(test_bacterial_leaf_blight_dir.glob(r"**/*.jpg")) + list(test_bacterial_leaf_blight_dir.glob(r"**/*.JPG"))
print("bacterial leaf blight : ",len(test_bacterial_leaf_blight))

test_leaf_smut_dir = Path("../tmp/rice_leaf_diseases_8/test/Leaf smut/")
test_leaf_smut = list(test_leaf_smut_dir.glob(r"**/*.jpg")) + list(test_leaf_smut_dir.glob(r"**/*.JPG"))
print("leaf smut : ",len(test_leaf_smut))

test_healthy_dir = Path("../tmp/rice_leaf_diseases_8/test/Healthy/")
test_healthy = list(test_healthy_dir.glob(r"**/*.jpg")) + list(test_healthy_dir.glob(r"**/*.JPG"))
print("healthy : ",len(test_healthy))

test_dataset = test_bacterial_leaf_blight + test_brown_spot + test_leaf_smut + test_healthy
print(len(test_dataset))
```
```
brown spot :  540
bacterial leaf blight :  540
leaf smut :  432
healthy :  540
2052
```

Using ImageDataGenerator
and add some image augmentation using parameters on ImageDataGenerator method
- width_shift_range
- height_shift_range
- shear_range
- horizontal_flip
- vertical_flip
- fill_mode 

```
import tensorflow as tf
 
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.25
)


test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)
 
train_images = train_generator.flow_from_dataframe(
    dataframe= train_df,
    x_col = 'Filepath',
    y_col = 'Label',
    target_size=(224,224),
    color_mode='rgb',
    shuffle=True,
    seef=42,
    class_mode="categorical",
    subset='training'

)
val_images = train_generator.flow_from_dataframe(
    dataframe= train_df,
    x_col = 'Filepath',
    y_col = 'Label',
    target_size=(224, 224),
    class_mode="categorical",
    color_mode='rgb',
    subset='validation'
)
  
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)
```
[![result-data-augmentation-implicitly.jpg](https://i.postimg.cc/mrHc8mZh/result-data-augmentation-implicitly.jpg)](https://postimg.cc/Y4tqSfWw)
## Model
```
own_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4, activation="softmax")
])

own_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
## Training
```
H = own_model.fit(
    train_images, 
    validation_data= val_images,
    epochs=15,
)
```
## Training Result
[![training-result-custom-model.jpg](https://i.postimg.cc/cHhRZFpT/training-result-custom-model.jpg)](https://postimg.cc/k6V6vvXt)

```
import matplotlib.pyplot as plt

plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title("Accuracy")
```
[![accuracy-own-model.jpg](https://i.postimg.cc/zvbKYq8S/accuracy-own-model.jpg)](https://postimg.cc/JH8GbVyG)
```
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.legend(['training','validation'])
plt.title("Loss")
```
[![loss-own-model.jpg](https://i.postimg.cc/7ZM7ks8c/loss-own-model.jpg)](https://postimg.cc/21y3dx5F)

## Result on Test Images
```
import numpy as np

pred = own_model.predict(test_images)
pred = np.argmax(pred,axis=1)

labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]
print(f'The first 10 predictions: {pred[:10]}')

from sklearn.metrics import accuracy_score
y_test = list(test_df.Label)
acc = accuracy_score(y_test,pred)
print(f'Accuracy on the test set: {acc * 100:.2f}%') 


from sklearn.metrics import classification_report
class_report = classification_report(y_test, pred, zero_division=1)
print(class_report)
```

```
The first 10 predictions: ['Healthy', 'Brown spot', 'Healthy', 'Leaf smut', 'Bacterial leaf blight', 'Healthy', 'Bacterial leaf blight', 'Leaf smut', 'Healthy', 'Leaf smut']
Accuracy on the test set: 99.71%
                       precision    recall  f1-score   support

Bacterial leaf blight       1.00      0.99      1.00       540
           Brown spot       0.99      1.00      1.00       540
              Healthy       1.00      1.00      1.00       540
            Leaf smut       1.00      1.00      1.00       432

             accuracy                           1.00      2052
            macro avg       1.00      1.00      1.00      2052
         weighted avg       1.00      1.00      1.00      2052
```
