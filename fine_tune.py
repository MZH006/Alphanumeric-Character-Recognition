from train_model import DigitCNN
import torch.utils.data.dataset
from torch.utils.data import DataLoader
from torchvision import transforms 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


import os
from PIL import Image
from PIL import ImageOps

class fine_tuning(torch.utils.data.Dataset):

    def __init__(self, transform = None):
        super().__init__()
        self.transform = transform
        self.image_paths = []
        self.labels = [] 
        for label_fol in os.listdir("my_digits"):
            full_path = os.path.join("my_digits", label_fol)

            for image in os.listdir(full_path):
                full_image_path = os.path.join(full_path, image)
                self.image_paths.append(full_image_path)
                self.labels.append(int(label_fol))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        opened_image = Image.open(image_path)
        opened_image = opened_image.convert("L")
        if self.transform:
            opened_image = self.transform(opened_image)

        return opened_image, label
    
    def __len__(self):
        return len(self.image_paths)
    

def get_dataloader(batch_size = 64, shuffle=True):
    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    my_dataset = fine_tuning(transform=train_transform)
    train_size = int(0.8 * len(my_dataset))
    test_size = len(my_dataset) - train_size
    train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_dataloader()
    model = DigitCNN()
    model.load_state_dict(torch.load("digit_model.pt"))
    criterion = nn.CrossEntropyLoss()
    # for param in model.conv.parameters():
    #     param.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)


    best_test_acc = 0.0
    epochs = 500
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()

        # Train Accuracy
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        train_acc = 100 * train_correct / train_total

        # Test Accuracy
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = 100 * test_correct / test_total

        # Logging
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy after epoch {epoch+1}: {test_acc:.2f}%")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        model.train()

        print(f"Test Accuracy after epoch {epoch+1}: {test_acc:.2f}%")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        scheduler.step(total_loss)
        torch.save(model.state_dict(), "fine_tuned_digit_model.pt")