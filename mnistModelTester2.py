import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Constants
WINDOWSIZEX = 680
WINDOWSIZEY = 480
BOUNDARYINC = 5
PREDICT = True
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Define the model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the PyTorch model
model = CNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Set the model to evaluation mode

# Initialize Pygame
pygame.init()
FONT = pygame.font.Font(None, 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

# Variables for tracking drawing
drawing = False
last_pos = (0, 0)
image_data = []

def preprocess_image(img_arr):
    img = cv2.resize(img_arr, (28, 28))
    img = cv2.bitwise_not(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32)

def predict_digit(image):
    processed_image = preprocess_image(image)
    with torch.no_grad():
        output = model(processed_image)
    digit = output.argmax(dim=1, keepdim=True).item()
    return digit

def get_bounding_box(img):
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def extract_and_preprocess(surface_array):
    gray_image = cv2.cvtColor(surface_array, cv2.COLOR_RGB2GRAY)
    inverted_image = cv2.bitwise_not(gray_image)

    # Extract the bounding box
    bounding_box_image = get_bounding_box(inverted_image)

    # Check if bounding box is valid
    if bounding_box_image.size == 0:
        print("No drawing detected")
        return None

    # Resize image
    resized_image = cv2.resize(bounding_box_image, (28, 28))
    return resized_image

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos

        if event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = event.pos
                pygame.draw.line(DISPLAYSURF, WHITE, last_pos, current_pos, 10)
                last_pos = current_pos

        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False

            # Convert the pygame surface to an array
            surface_array = pygame.surfarray.array3d(DISPLAYSURF)

            # Extract and preprocess the image
            preprocessed_image = extract_and_preprocess(surface_array)
            if preprocessed_image is None:
                continue

            plt.imshow(preprocessed_image, cmap='gray')
            plt.title('Preprocessed Image')
            plt.show()

            digit = predict_digit(preprocessed_image)

            # Display the predicted digit
            text_surface = FONT.render(LABELS[digit], True, RED, WHITE)
            text_rect = text_surface.get_rect()
            # Adjust the position to be relative to the drawing box
            text_rect.topleft = (last_pos[0] + 10, last_pos[1] + 10)
            DISPLAYSURF.blit(text_surface, text_rect)

    pygame.display.update()
