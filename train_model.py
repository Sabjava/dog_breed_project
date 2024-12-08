import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, device):
    """
    Test the model on the test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Loss: {running_loss / len(test_loader)}")
    print(f"Accuracy: {100 * correct / total}%")

def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Training Loss: {running_loss / len(train_loader)}")

def net(optimizer_type, learning_rate, device):
    """
    Initialize and return the model.
    This function allows using different optimizers.
    """
    # Use a pre-trained model (ResNet18 as an example)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 133)  # Assuming 133 classes (e.g., dog breeds)

    # Send the model to the device (GPU/CPU)
    model = model.to(device)
    
    return model, optimizer_type, learning_rate

def create_data_loaders(data_dir, batch_size):
    """
    Create data loaders for training and validation.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Assuming data_dir contains 'train' and 'valid' directories
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'valid'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model and optimizer
    model, optimizer_type, learning_rate = net(args.optimizer, args.learning_rate, device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size)

    # Select optimizer based on the optimizer_type argument
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Train the model
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train(model, train_loader, criterion, optimizer, device)

        # Test the model after each epoch
        test(model, valid_loader, criterion, device)

    # Save the trained model
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters to pass from SageMaker
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help="Optimizer to use.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data', help="Path to input data.")
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model', help="Directory to save the trained model.")

    args = parser.parse_args()

    main(args)
