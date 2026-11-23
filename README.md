# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import necessary libraries for PyTorch, torchvision, visualization, and evaluation metrics.

### STEP 2: 

Define image transformations: convert tensors and normalize dataset between -1 and 1.


### STEP 3: 

Load MNIST training and testing datasets with transformations and enable downloading automatically.


### STEP 4: 

Create DataLoader objects for training and testing datasets with batch processing.


### STEP 5: 

Define CNNClassifier class with convolution, pooling, and fully connected layers.


### STEP 6: 

Implement forward function using ReLU activations, pooling, and flattening before classification.

### STEP 7: 

Initialize CNN model, move to GPU if available, and display summary.

### STEP 8:

Define cross-entropy loss function and Adam optimizer with learning rate 0.001.


### STEP 9:

Train model for epochs: forward pass, compute loss, backpropagate, and update parameters.


### STEP 10: 

Test model: predict outputs, calculate accuracy, store predictions, and generate confusion matrix.

### STEP 11: 

Visualize confusion matrix using heatmap and display classification report with precision metrics.


### STEP 12: 

Predict single image: load from dataset, infer, and display actual-predicted labels.

## PROGRAM

### Name: AKSHAY KARTHICK ASR

### Register Number: 212224230015

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # 16 filters, 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, 1) # 32 filters, 3x3
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model(model, train_loader, num_epochs=5)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Test Accuracy: {correct/total:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))

test_model(model, test_loader)

def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f'Actual: {label} | Predicted: {predicted.item()}')
    plt.axis("off")
    plt.show()

predict_image(model, image_index=80, dataset=test_dataset)

```

### OUTPUT

## Training Loss per Epoch

<img width="211" height="115" alt="Screenshot 2026-11-23 at 1 40 51 PM" src="https://github.com/user-attachments/assets/f15cfc5e-bbc9-4602-be82-87acb6a0a682" />


## Confusion Matrix

<img width="675" height="539" alt="Screenshot 2026-11-23 at 1 41 08 PM" src="https://github.com/user-attachments/assets/cce10f59-2a32-45d4-bbdc-23fa845af365" />


## Classification Report
<img width="450" height="313" alt="Screenshot 2026-11-23 at 1 41 22 PM" src="https://github.com/user-attachments/assets/2077204b-2e4e-4f80-aee7-bb8dd2d3c8d8" />

### New Sample Data Prediction
<img width="412" height="438" alt="Screenshot 2026-11-23 at 1 41 40 PM" src="https://github.com/user-attachments/assets/206830ac-b76c-46b3-8981-2ec65edc99f9" />


## RESULT
A convolutional neural network (CNN) classification model for the given dataset was developed and executed successfully.
