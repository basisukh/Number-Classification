import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

class DrawApp:
    def __init__(self, root):
        #create the application window
        self.root = root
        self.root.title("Draw a Number")

        #create the canvas
        self.canvas = Canvas(self.root, width=200, height=200, bg="white")
        self.canvas.pack()
        #set left click to draw
        self.canvas.bind("<B1-Motion>", self.paint) 
        #create clear button
        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas) 
        self.button_clear.pack()
        #create predict button
        self.button_predict = tk.Button(self.root, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Load the pre-trained MNIST model
        self.model = tf.keras.models.load_model('mnist_model.h5')  # Replace with your model path

    def paint(self, event):
        #draw on the canvas
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.line([x1, y1, x2, y2], fill=0, width=10)

    def clear_canvas(self):
        #clear the canvas
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill=255)

    def preprocess_image(self):
        # Resize to 28x28 pixels as required by MNIST model
        image = self.image.resize((28, 28))
        # Invert colors: white (background) to black, black (number) to white
        image = ImageOps.invert(image)
        # Normalize to 0-1
        image = np.array(image) / 255.0
        # Reshape to (1, 28, 28, 1)
        image = image.reshape(1, 28, 28, 1)
        return image

    def predict(self):
        #use the predict method to guess the number
        image = self.preprocess_image()
        prediction = self.model.predict(image)
        predicted_number = np.argmax(prediction)
        #print predicted number
        print(f"Predicted Number: {predicted_number}")
        self.show_prediction(predicted_number)

    def show_prediction(self, number):
        #show the predicted number
        prediction_window = tk.Toplevel(self.root)
        prediction_window.title("Prediction")
        tk.Label(prediction_window, text=f"Predicted Number: {number}", font=("Helvetica", 24)).pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
