import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_set = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), 
            nn.ReLU(),
            nn.MaxPool2d(2),         
            nn.Conv2d(32, 64, 3, 1), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 47)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_test_acc = 0.0
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        

        print(f"Train Accuracy after epoch {epoch+1}: {100 * correct / total:.2f}%")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        scheduler.step(total_loss)
        torch.save(model.state_dict(), "digit_model.pt")


    test_set = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    if epoch == 0 or test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        torch.save(model.state_dict(), "digit_model.pt")
        print('best model saved')

