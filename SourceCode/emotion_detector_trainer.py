import numpy as np
np.object = np.object_          # Resolve deprecation warnings for numpy data types
np.bool = np.bool_
np.int = np.int32
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

# Directory paths for training and testing images
TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'

# Function to create a DataFrame with image paths and labels
def create_data_frame(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for image_name in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,image_name))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

# Creating DataFrames for training and testing data
train =pd.DataFrame()
train['image'], train['label'] = create_data_frame(TRAIN_DIR)

test = pd.DataFrame()
test['image'], test['label'] = create_data_frame(TEST_DIR)

# Function to extract features from images
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image,grayscale = True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

# Extracting features for training and testing datasets
train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

# Normalizing pixel values to range [0, 1]
x_train = train_features/255.0
x_test = test_features/255.0

# Encoding labels to integers
le = LabelEncoder()
le.fit(train['label'])

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

# Converting integer labels to one-hot encoded format
y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)

# Callbacks for early stopping and saving the best model
call_back = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
best_model_file = "best_model_1.keras"
best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)

# Building the Convolutional Neural Network (CNN) model
model = Sequential()
# Convolutional layers with pooling and dropout for regularization
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu', input_shape = (48,48,1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# Fully connected layers with dropout
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
# Output layer with softmax activation for multi-class classification
model.add(Dense(7, activation = 'softmax'))

# Compiling the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

# Training the model
history = model.fit(x = x_train, y = y_train, batch_size = 128, epochs = 100, validation_data = (x_test, y_test), callbacks=[call_back, best_model])

# Plotting training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc , 'r', label="Train accuracy")
plt.plot(epochs, val_acc , 'b', label="Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and validation accuracy')
plt.legend(loc='lower right')
plt.show()

# Plotting training and validation loss
fig2 = plt.figure(figsize=(14,7))
plt.plot(epochs, loss , 'r', label="Train loss")
plt.plot(epochs, val_loss , 'b', label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and validation Loss')
plt.legend(loc='upper right')
plt.show() 