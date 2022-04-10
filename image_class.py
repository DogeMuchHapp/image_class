#image classifier

#lib
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#load
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#data types
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

#shape array
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

#image array
index = 4345
x_train[index]

#image as pic
img = plt.imshow(x_train[index])

#image label
print('The image label is:', y_train[index])

#image class
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#print image class
print('The image class is:', classification[y_train[index][0]])

#convert
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#print label
print(y_train_one_hot)

#print new label
print('The one hot label is:', y_train_one_hot[index])

#normalize
x_train = x_train / 255
x_test = x_test / 255

x_train[index]

#new cell
from keras import activations
#arch
model = Sequential()

#first layer
model.add( Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)) )

#pool
model.add(MaxPooling2D(pool_size = (2,2)))

#add another layer
model.add( Conv2D(32, (5,5), activation='relu') )

#add another pool
model.add(MaxPooling2D(pool_size = (2,2)))

#flat
model.add(Flatten())

#add layer 1000 neurons
model.add(Dense(1000, activation='relu'))

#dropout
model.add(Dropout(0.5))

#add layer 500 neurons
model.add(Dense(500, activation='relu'))

#dropout
model.add(Dropout(0.5))

#add layer 250 neurons
model.add(Dense(250, activation='relu'))

#add layer 10 neurons
model.add(Dense(10, activation='softmax'))

#new cell
#complile
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
              
#train
hist = model.fit(x_train, y_train_one_hot,
                 batch_size = 256,
                 epochs = 10,
                 validation_split= 0.2)

#eval using test
model.evaluate(x_test, y_test_one_hot)[1]

#visualization
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show

#model loss visual
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show

#show image
new_image = plt.imread('image.jpg')
img = plt.imshow(new_image)

#resize
from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))
img = plt.imshow(resized_image)

#predict
predictions = model.predict(np.array([resized_image]))
#show
predictions

#sort
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
  for j in range(10):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp

#show
print(list_index)

#print first 5 most likely
for i in range(5):
  print(classification[list_index[i]], ':', predictions[0][list_index[i]] * 100, '%')

if list_index[0] == 0:
  print("BIG BIRD IN THE SKY")
if list_index[0] == 1:
  print("CAR GO VROOM VROOM!")
if list_index[0] == 2:
  print("LITERAL BIRD IN THE SKY")
if list_index[0] == 3:
  print("CURIOSITY KILLED THE CAT")
if list_index[0] == 4:
  print("JUST GO STARE INTO SOME HEADLIGHTS")
if list_index[0] == 5:
  print("WOOF WOOF, MANS BEST FRIEND")
if list_index[0] == 6:
  print("HOP HOP, NOT A BUNNY")
if list_index[0] == 7:
  print("BETTING IS ILLEGAL FOR THOSE UNDER THE AGE OF 18")
if list_index[0] == 8:
  print("SOME PEOPLE GET SEASICK")
if list_index[0] == 9:
  print("BIG CAR GO SINGLE VROOM")
