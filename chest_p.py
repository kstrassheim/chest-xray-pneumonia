import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, TensorBoard
from time import time
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

PATH ='C:\\projects\\chest_xray'

batchSize = 32
imgSize = 150
train_data_gen = ImageDataGenerator().flow_from_directory(
    directory=os.path.join(PATH, 'train'),
    target_size=(imgSize, imgSize),
    class_mode='binary',
    batch_size=batchSize)

val_data_gen = ImageDataGenerator().flow_from_directory(
    directory=os.path.join(PATH, 'val'),
    target_size=(imgSize, imgSize),
    class_mode='binary',
    batch_size=batchSize)

test_data_gen = ImageDataGenerator().flow_from_directory(
    directory=os.path.join(PATH, 'test'),
    target_size=(imgSize, imgSize),
    class_mode='binary',
    batch_size=batchSize)

nn = Sequential()
# input shape = image shape + 3 colors on 3rd dimension
nn.add(Conv2D(32,3,padding='same', activation='relu',input_shape=(imgSize,imgSize,3)))
nn.add(MaxPool2D())
nn.add(Dropout(0.2))
nn.add(BatchNormalization())
nn.add(Conv2D(64, 3, padding='same',activation='relu'))
nn.add(MaxPool2D())
nn.add(Dropout(0.2))
nn.add(BatchNormalization())
nn.add(Conv2D(128, 3, padding='same',activation='relu'))
nn.add(MaxPool2D())
nn.add(Dropout(0.2))
nn.add(BatchNormalization())
nn.add(Conv2D(256, 3, padding='same',activation='relu'))
nn.add(MaxPool2D())
nn.add(Dropout(0.2))
nn.add(BatchNormalization())
nn.add(Flatten())
nn.add(Dense(128,activation='relu'))
nn.add(Dense(1,activation='sigmoid'))
nn.summary()
nn.compile(optimizer='rmsprop', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
nn.fit(train_data_gen, validation_data=val_data_gen, epochs=3, batch_size=batchSize, validation_batch_size=16, callbacks=[tensorboard])


evalResult = nn.evaluate(test_data_gen)
print("Loss of the model is - " , evalResult[0])
print("Accuracy of the model is - " , evalResult[1]*100 , "%")
#nn.save('saved_model/my_model') 
pred = nn.predict(test_data_gen)
ypred = np.concatenate(np.uint(np.round(nn.predict(test_data_gen))))
ytrue = np.uint(np.concatenate([test_data_gen[i][1] for i in range(0, len(test_data_gen))]))
confusionmatrix = confusion_matrix(ytrue, ypred)
sns.heatmap(confusionmatrix, annot=True, fmt="d", xticklabels=['NORMAL-True', 'PNEUMONIA-True'], yticklabels=['NORMAL-Pred', 'PNEUMONIA-Pred'])

