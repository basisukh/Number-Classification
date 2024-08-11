import tensorflow as tf
from tensorflow.keras.models import load_model

print("TensorFlow version:", tf.__version__)
model = load_model('mnist_model.h5')
print("Model loaded successfully!")
