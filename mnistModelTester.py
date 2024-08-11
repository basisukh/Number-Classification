from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist

# Load and preprocess the custom image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img = np.invert(np.array(img))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

image_path = '/Users/akalsukhbasi/Desktop/Java Files/csa/src/advtopics/test4.png'
custom_img = preprocess_image(image_path)

# Display the preprocessed custom image
plt.imshow(custom_img.reshape(28, 28), cmap='gray')
plt.title('Preprocessed Custom Image')
plt.show()

# Load the model
model = load_model('mnist_model.h5')

# Predict the digit and print the prediction vector
prediction = model.predict(custom_img)
print(f'Prediction Vector: {prediction}')
digit = np.argmax(prediction)
print(f'Predicted Digit: {digit}')

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255.0

# Visualize the predictions for some MNIST images of digit 4
digit_4_indices = np.where(y_test == 4)[0]
for i in range(5):
    img = x_test[digit_4_indices[i]].reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    
    plt.imshow(x_test[digit_4_indices[i]].reshape(28, 28), cmap='gray')
    plt.title(f'MNIST Image of 4, Predicted: {predicted_digit}')
    plt.show()

# Compare with the prediction vector of custom image
for idx in digit_4_indices[:5]:
    mnist_img = x_test[idx].reshape(1, 28, 28, 1)
    mnist_prediction = model.predict(mnist_img)
    mnist_digit = np.argmax(mnist_prediction)
    print(f'MNIST Image Prediction Vector: {mnist_prediction}, Predicted Digit: {mnist_digit}')
