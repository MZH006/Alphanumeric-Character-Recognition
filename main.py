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

model = DigitCNN()
model.load_state_dict(torch.load("digit_model.pt"))
model.eval()
emnist_data = EMNIST(root='./data', split='balanced', download=True)
class_mapping = emnist_data.classes


def render_prediction(surface, prediction):
    pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(0, 0, 200, 30))
    text_surface = font.render(f"Prediction: {prediction}", True, (0, 255, 0))
    surface.blit(text_surface, (10, 5))


def preprocess_and_predict(surface):
    data = pygame.surfarray.array3d(surface)
    grayscale = np.dot(data, [0.2989, 0.5870, 0.1140])
    grayscale = np.transpose(grayscale)

    image = Image.fromarray(grayscale).convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = Image.eval(image, lambda x: 255 - x)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
    
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


