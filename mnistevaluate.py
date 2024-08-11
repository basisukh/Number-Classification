from keras.datasets import mnist
import numpy as np
from keras.models import load_model

# Load the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = load_model('mnist_model.h5')


# Preprocess the test data
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Model accuracy on MNIST test set: {accuracy * 100:.2f}%')
