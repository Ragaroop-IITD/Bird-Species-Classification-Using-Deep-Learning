import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from sklearn.metrics import f1_score
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


dataPath = sys.argv[1]
trainStatus = sys.argv[2]
modelPath = sys.argv[3] if len(sys.argv) > 3 else "error"


# Set the seed for reproducibility
torch.manual_seed(0)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
num_epochs = 35
train_batch_size = 32  
test_batch_size=1
learning_rate = 0.0001  
num_classes = 10

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Increased image size for better feature extraction
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])


# Enhanced CNN Model
class BirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth convolutional block
            nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Sixth convolutional block
            nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 3 * 3, 2048),  # Adjusted for 224x224 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x




# Initialize model, loss function, and optimizer
model = BirdClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Training function with progress tracking and loss/accuracy logging
def train_model(model, train_loader, test_loader, train_eval_loader, num_epochs):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(".", end="")
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
        
        # Evaluate on validation set
        epoch_val_loss, epoch_val_acc = evaluate_model(model, test_loader)
        epoch_train_loss,epoch_train_acc= evaluate_model(model, train_eval_loader)
        
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), modelPath)

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    return running_loss/len(data_loader), accuracy




def test_model(model, test_loader, output_csv='bird.csv'):
    model.eval()  
    total = 0
    results = []
    with torch.no_grad():  
        for images, _ in test_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            # Get predictions
            predicted=torch.argmax(outputs,dim=1)
            results.extend(predicted.cpu().numpy())


    # Write only predicted labels to a CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        for label in results:
            writer.writerow([label])


# Evaluation function
def evaluate_model1(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    return accuracy,f1_macro,f1_micro




if trainStatus == "train":
    
    # Load and split dataset
    data_path = dataPath
    dataset = ImageFolder(root=data_path, transform=transform)
    targets = np.array(dataset.targets)


    # Initialize StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Get indices for train and test sets
    for train_index, test_index in sss.split(np.arange(len(targets)), targets):
        train_indices, test_indices = train_index, test_index

    # Create subsets and data loaders
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_eval_loader= DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    print("Starting training...")
    # Train the model
    train_model(model, train_loader,test_loader,train_eval_loader, num_epochs)

    print("\nEvaluating model...")
    a,b,c=evaluate_model1(model,test_loader)
    print(f"Accuracy:{a} f1_macro:{b} f1_micro:{c}")
else:
    print("testing")
    dataset = ImageFolder(root=dataPath, transform=transform)
    test_loader1 = DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
    model.load_state_dict(torch.load(modelPath,weights_only=False))
    test_model(model,test_loader1)
    

