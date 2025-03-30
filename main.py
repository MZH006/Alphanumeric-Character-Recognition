from train_model import DigitCNN
from canvas import run_canvas
import torch

import pygame
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("Arial", 24)

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
import torch.nn.functional as F

import cv2
from PIL import ImageOps
from scipy import ndimage
import matplotlib.pyplot as plt

model = DigitCNN()
model.load_state_dict(torch.load("digit_model.pt"))
model.eval()
emnist_data = EMNIST(root='./data', split='digits', download=True)
class_mapping = emnist_data.classes 


def render_prediction(surface, prediction):
    pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(0, 0, 200, 30))
    text_surface = font.render(f"Prediction: {prediction}", True, (0, 255, 0))
    surface.blit(text_surface, (10, 5))


def preprocess_and_predict(surface):

    data = pygame.surfarray.array3d(surface)
    grayscale = np.dot(data, [0.2989, 0.5870, 0.1140])
    grayscale = np.transpose(grayscale) 

    image = Image.fromarray(grayscale).convert("L")
    image = ImageOps.invert(image)

    image = image.point(lambda p: 255 if p > 50 else 0)

    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    image = image.resize((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    upper_left = ((28 - 20) // 2, (28 - 20) // 2)
    canvas.paste(image, upper_left)

    np_image = np.array(canvas)
    cy, cx = ndimage.center_of_mass(np_image)
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    np_image = cv2.warpAffine(np_image, M, (28, 28), borderValue=0)
    
    np_image = 255 - np_image

    # Force white stroke and black background (binarize)
    np_image[np_image < 100] = 0
    np_image[np_image >= 100] = 255
    
    final_image = Image.fromarray(np_image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = transform(final_image).unsqueeze(0)

    emnist_digits = EMNIST(root='./data', split='digits', train=True, download=True)
    zeros = [img for img, label in emnist_digits if label == 0][:1]  # one zero image

    plt.subplot(1, 2, 1)
    plt.imshow(tensor.squeeze().numpy(), cmap='gray')
    plt.title("Your Drawn 0")

    plt.subplot(1, 2, 2)
    plt.imshow(np.array(zeros[0]), cmap='gray')
    plt.title("EMNIST 0")

    plt.tight_layout()
    plt.show()

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        print("Class Probabilities:", probs.squeeze().numpy())
        pred = torch.argmax(probs, dim=1).item()

    return pred


running = True

while running:
    surface = run_canvas()
    prediction_index = preprocess_and_predict(surface)
    predicted_char = class_mapping[prediction_index]

    render_prediction(surface, predicted_char)
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                waiting = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    waiting = False 
                elif event.key == pygame.K_q:  
                    running = False
                    waiting = False

pygame.quit()


