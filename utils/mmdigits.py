import os
import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MMDataLoader(Dataset):
    def __init__(self, dataset_dir, desired_width=64, desired_height=64, top_crop=5, bottom_crop=20, left_crop=50, right_crop=50):
        self.images, self.labels = self.load_dataset(dataset_dir, desired_width, desired_height, top_crop, bottom_crop, left_crop, right_crop)

    def load_dataset(self, dataset_dir, desired_width, desired_height, top_crop, bottom_crop, left_crop, right_crop):
        images = []
        labels = []
        for class_label in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_label)
            if os.path.isdir(class_dir):
                label = int(class_label)
                for filename in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = image[top_crop:image.shape[0]-bottom_crop, left_crop:image.shape[1]-right_crop]
                    image = cv2.resize(image, (desired_width, desired_height))
                    images.append(image)
                    labels.append(label)

        images = np.array(images)
        labels = np.array(labels)
        images = images.astype(np.float32) / 255.0  # Normalize to [0, 1]
        images = images.reshape(-1, desired_width * desired_height)  # Flatten images

        return torch.tensor(images), torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DigitClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DigitClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds * 100
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


def evaluate_model(model, test_loader):
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


# Example usage:

# Define directories and dataset path
dataset_dir = "./data/MM_digits"

# Load the dataset
data_loader = MMDataLoader(dataset_dir)
X = data_loader.images
y = data_loader.labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model parameters
input_size = X.shape[1]
hidden_sizes = [128, 64]
output_size = 10  # Number of classes (digits 0-9)

# Initialize the model, loss function, and optimizer
model = DigitClassifier(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=10)

# Evaluate the model on test data
evaluate_model(model, test_loader)

checkpoint = {
    'input_size': input_size,  
    'output_size': output_size,  
    'hidden_layers': hidden_sizes,  
    'state_dict': model.state_dict()  
}

model_path = "./models/mmdigit_classifier_model.pth"
torch.save(checkpoint, model_path)



