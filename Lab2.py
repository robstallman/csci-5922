# %% Imports
import copy
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
from torchvision import datasets
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
############
## Part 1 ##
############

# %% ----- Selecting a Difficult Dataset -----

# Download training and testing sets from CIFAR100
raw_tfms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
training_data = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=raw_tfms
)

test_data = datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=raw_tfms
)

# %% ----- Data Preprocessing -----

# Concatenate training and testing data along dimension 1 (x axis of each image)
x_train = torch.concat([sample[0] for sample in training_data], dim=1)
x_test = torch.concat([sample[0] for sample in test_data], dim=1)

# Combine training and testing data
X = torch.concat((x_train, x_test), dim=1)

# Find mean and standard deviation for each channel across all samples
mu = X.mean(dim=(1,2))
sigma = X.std(dim=(1,2))
print(f"CIFAR-100 means: {mu}")
print(f"CIFAR-100 standard deviations: {sigma}")

# Transforms for normalizing data – we'll leave the training data untouched otherwise
tfms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mu, sigma),
])

# Load datasets for training and testing, applying our normalization transformation to both
train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=tfms)
test_set  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=tfms)

# Divide datasets into mini-batches
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

classes = train_set.classes
print(f"Training dataset contains {len(classes)} classes")

# %% ----- Evaluating the Dataset Difficulty: Definitions -----

# Define sigmoid activation function
def sigmoid(z: torch.tensor) -> torch.tensor:
    return 1.0/(1.0+torch.exp(-z))

# Define softmax activation function
def softmax(z: torch.tensor) -> torch.tensor:
    y = torch.exp(z)
    y_tot = y.sum(dim=1).unsqueeze(1)
    return y / y_tot

# Define cross-entropy loss
def cross_entropy_loss(logits: torch.tensor, y: torch.tensor) -> torch.tensor:
    # Get a tensor of predicted probabilities for each target class in the mini-batch
    probs = logits[torch.arange(logits.size(0)), y]
    
    # Calculate the loss for each example
    loss = -1 * torch.log(probs)

    # Return average loss over the examples
    return loss.mean()

# Define a simple two-layer network
class TwoLayerNetwork(nn.Module):
    def __init__(self, input_size: int = 3072, hidden_size: int = 3072, n_classes: int = 100):
        super(TwoLayerNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        # Define all layers in the model
        # layer 1
        self.linear1 = nn.Linear(self.input_size, hidden_size)
        # layer 2
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # layer 3
        self.linear3 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.tensor):
        # Build the feed forward structure
        x = x.view(x.size(0), -1)     # flatten
        linear1 = self.linear1(x)
        act1 = sigmoid(linear1)
        linear2 = self.linear2(act1)
        act2 = sigmoid(linear2)
        linear3 = self.linear3(act2)
        output = softmax(linear3)
        return output

    def backward(self, loss: torch.tensor, lr: float = 0.1):
        # Reset parameter gradients
        self.linear1.weight.grad = None
        self.linear1.bias.grad = None
        self.linear2.weight.grad = None
        self.linear2.bias.grad = None
        self.linear3.weight.grad = None
        self.linear3.bias.grad = None

        # Update gradients
        loss.backward()

        # Update parameters
        self.linear1.weight.data -= lr * self.linear1.weight.grad
        self.linear1.bias.data -= lr * self.linear1.bias.grad
        self.linear2.weight.data -= lr * self.linear2.weight.grad
        self.linear2.bias.data -= lr * self.linear2.bias.grad
        self.linear3.weight.data -= lr * self.linear3.weight.grad
        self.linear3.bias.data -= lr * self.linear3.bias.grad


# Define a function to evaluate a single epoch
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train: bool = True,
    lr: float = 0.1,
) -> tuple[torch.tensor, torch.tensor]:
    # Make sure we're on the correct device
    model = model.to(device)

    # Set the model mode – either training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # Initial values for tracking loss and accuracy over the epoch
    loss, n_correct, n_total = 0,0,0

    # Set context based on model mode
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        # Iterate through the batches in the loader
        for xb, yb in loader:
            # Transfer to device
            xb, yb = xb.to(device), yb.to(device)

            # -- Forward pass -- #
            # Get probabilities for each class
            logits = model(xb)

            # Get average loss over all the examples in the mini-batch
            loss = cross_entropy_loss(logits, yb)

            # -- Backward pass -- #
            if train:
                model.backward(loss, lr)

            # Update running counts for loss and accuracy
            loss += loss.item()
            predictions = logits.argmax(dim=1)
            n_correct += (predictions == yb).sum()
            n_total += yb.shape[0]

    # Calculate accuracy for this epoch
    acc = n_correct / n_total

    # Return loss and accuracy for this epoch
    return loss.item(), acc


# Define function for training
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: int = 0.1,
    n_epochs: int = 1000,
    print_every: int = 50,
    early_stopping: bool = True,
    patience: int = 5000,
    min_delta: float = 1e-2,
):
    # Lists for storing epochs, losses, and accuracies
    epochs = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Early stopping setup
    best_test_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch = 1
    no_improvement_count = 0

    # Start training
    for epoch in range(n_epochs):
        # Training over epoch
        train_loss, train_acc = run_epoch(
            model=model, loader=train_loader, device=device, train=True, lr=lr
        )

        # Testing over epoch
        test_loss, test_acc = run_epoch(
            model=model, loader=test_loader, device=device, train=False, lr=lr
        )

        # Save losses and accuracies for this epoch
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Print the losses periodically
        if epoch % print_every == (print_every - 1):
            print(f'Epoch {epoch+1}/{n_epochs}\n'
                  f'--------------------------\n'
                  f'Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}\n'
                  f'Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}\n')

        # Early stopping logic
        if early_stopping:
            # Improvement means test_loss got smaller by at least min_delta
            if test_loss < best_test_loss - min_delta:
                best_test_loss = test_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}\n"
                          f"Best epoch was {best_epoch} with test loss {best_test_loss:.4f}")
                    break

        # Reload best model
        if early_stopping:
            model.load_state_dict(best_model_weights)

        # Gather epochs, losses, accuracies to return
        training_curve = {
            'epochs': epochs,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }

    return model, training_curve


# Define a function for visualizing training curves
def plot_loss_acc(training_curve):
    epochs = training_curve['epochs']
    train_losses = training_curve['train_losses']
    test_losses = training_curve['test_losses']
    train_accuracies = training_curve['train_accuracies']
    test_accuracies = training_curve['test_accuracies']

    # Create a figure and subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    # Plot loss on the first subplot
    ax[0].plot(epochs, train_losses, label='Train set loss')
    ax[0].plot(epochs, test_losses, label='Test set loss')
    ax[0].set_title('Training Loss Over Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)
    ax[0].legend()

    # Plot accuracy on the second subplot
    ax[1].plot(epochs, train_accuracies, label='Train set accuracy')
    ax[1].plot(epochs, test_accuracies, label='Test set accuracy')
    ax[1].set_title('Training Accuracy Over Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# Define a function to save models
def save_model(model: nn.Module, name: str):
    # Create a directory for models if it doesn't yet exist
    if not os.path.exists("models"):
        os.mkdir("models")

    # Save the model to the directory
    filepath = os.path.join("models", f"{name}.pt")
    torch.save(model.state_dict(), filepath)

# %% ----- Evaluating the Dataset Difficulty: Training -----
# Create an instance of the model
model = TwoLayerNetwork().to(device)

# Train the model
model, training_curve = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    lr=0.1,
    n_epochs=6,
    print_every=2,
)

# Save the trained model
torch.save(model.state_dict(), "baseline_shallow_model.pt")

plot_loss_acc(training_curve)


# %% ----- Building a Baseline Deep Network: Definitions -----


# Define ReLU activation function
def ReLU(z: torch.tensor) -> torch.tensor:
    return torch.clamp(z, min=0)


# Define tanh activation function
def tanh(x: torch.tensor) -> torch.tensor:
    pos_exp = torch.exp(x)
    neg_exp = torch.exp(-x)
    return (pos_exp - neg_exp) / (pos_exp + neg_exp)


# Define a baseline network for deep learning
class BaselineDeepNetwork(nn.Module):
    def __init__(self, input_size: int = 3072, n_classes: int = 100):
        super(BaselineDeepNetwork, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes

        # -- Layer Definitions -- #
        # layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 2
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=0
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 3
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # layer 4
        # After first three layers we're left with [N_examples, 384, 4, 4]
        self.linear4 = nn.Linear(384 * 4 * 4, 1024)

        # layer 5
        self.linear5 = nn.Linear(1024, n_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Layer 1
        x = self.conv1(x)
        x = sigmoid(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = sigmoid(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = sigmoid(x)
        x = self.pool3(x)

        # Layer 4
        x = x.view(x.size(0), -1)  # flatten
        x = self.linear4(x)
        x = sigmoid(x)

        # Layer 5
        x = self.linear5(x)
        output = softmax(x)

        return output

    def backward(self, loss: torch.tensor, lr: float = 0.1) -> None:
        # Reset parameter gradients
        # NOTE: No learnable parameters for activation functions or pooling layers
        self.conv1.weight.grad = None
        self.conv1.bias.grad = None
        self.conv2.weight.grad = None
        self.conv2.bias.grad = None
        self.conv3.weight.grad = None
        self.conv3.bias.grad = None
        self.linear4.weight.grad = None
        self.linear4.bias.grad = None
        self.linear5.weight.grad = None
        self.linear5.bias.grad = None

        # Update gradients
        loss.backward()

        # Update parameters
        self.conv1.weight.data -= lr * self.conv1.weight.grad
        self.conv1.bias.data -= lr * self.conv1.bias.grad
        self.conv2.weight.data -= lr * self.conv2.weight.grad
        self.conv2.bias.data -= lr * self.conv2.bias.grad
        self.conv3.weight.data -= lr * self.conv3.weight.grad
        self.conv3.bias.data -= lr * self.conv3.bias.grad
        self.linear4.weight.data -= lr * self.linear4.weight.grad
        self.linear4.bias.data -= lr * self.linear4.bias.grad
        self.linear5.weight.data -= lr * self.linear5.weight.grad
        self.linear5.bias.data -= lr * self.linear5.bias.grad


# %% ----- Building a Baseline Deep Network: Training -----
# Create an instance of the model
model = BaselineDeepNetwork().to(device)

# Train the model
model, training_curve = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    lr=0.1,
    n_epochs=6,
    print_every=2,
)

# Save the trained model
torch.save(model.state_dict(), "baseline_deep_model.pt")

plot_loss_acc(training_curve)

# %%
