from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

from draw import draw_graphics

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# plt.imshow(train_images[0],cmap=plt.cm.binary)
# plt.show()
#
# print(train_labels[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

draw_graphics(H)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)