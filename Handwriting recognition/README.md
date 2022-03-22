~~~Python
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from IPython.display import display 
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist # Handwritten digit database
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Check data
display(x_train.shape)
display(y_train.shape)
display(x_test.shape)
display(y_test.shape)

display(y_test)
#-----

# Reshape the training data to 28 by 28 matrix and Normalize the data to 0~1 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255
#-----
# Create a neural network
model = Sequential() # One of the two models of Keras.
model.add(layers.InputLayer(input_shape=(28, 28, 1))) # Single layer
'''model.add(layers.Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1))) # CNN newral network
model.add(layers.MaxPooling2D(pool_size=(2, 2)))'''
model.add(layers.Flatten()) # Convert the layer to a 1 dimensional array
model.add(layers.Dense(128, activation='relu')) # First danse layer has 64 neurons.
#model.add(layers.Dropout(0.2))
#model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # Last layer is a softmax layer that has 10 neurons.
#-----

# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2) # "Fit" the model to the training data:
#-----

model.summary()

# Evaluate accuracy
test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
print('\nTest accuracy:', test_acc)
#-----

# Make predictions
image_index = 1
predictions = model.predict(x_test)
print(predictions[image_index])
print('This image is very likely to be : {}'.format(np.argmax(predictions[image_index])))

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.show()
#-----

def show_data_by_text(arr):
    plt.imshow(arr, cmap=plt.cm.binary) # Show binary data
    
    reshape_data = arr.reshape(-1, )
    for index, data in enumerate(reshape_data):
        print('{:3d}'.format(data), end='') # {:3d} = Calculate to 3 decimal places.
        if index % 28 == 27: # 마지막 column이면 줄바꿈
            print()

#show_data(x_train[image_index]) # Check the learned data
~~~
## Output
![image](https://user-images.githubusercontent.com/67142421/159577893-44b70daf-aae2-42a5-b585-7dc096991099.png)
