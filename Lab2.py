# %% Setup
# 3rd party imports
import copy
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
import typing
import wandb
from datetime import datetime
from dotenv import load_dotenv
from torchvision import datasets
from torch.utils.data import DataLoader

# Set up W&B
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
project_name = "CSCI5922_Lab2"
entity = os.getenv("WANDB_ENTITY")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# Run on Colab
if os.getcwd() == "/content":
    from google.colab import drive

    drive.mount("/content/drive")
    root_path = "/content/drive/My Drive/Colab Notebooks/CSCI 5922/"
else:
    root_path = os.getcwd()

# Run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Part 1
#####################
## Baseline Models ##
#####################

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
logging.info(f"CIFAR-100 means: {mu}")
logging.info(f"CIFAR-100 standard deviations: {sigma}")

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
batch_size = 1
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

classes = train_set.classes
logging.info(f"Training dataset contains {len(classes)} classes")

# %% ----- Evaluating the Dataset Difficulty: Definitions -----

# Define sigmoid activation function
def sigmoid(z: torch.tensor) -> torch.tensor:
    return 1.0/(1.0+torch.exp(-z))

# Define softmax activation function
def softmax(z: torch.tensor) -> torch.tensor:
    # Subtract maximum value from input for numerical stability
    maxvals, _ = z.max(dim=1)
    z -= maxvals.view(z.size(0), -1)

    # Normal softmax
    y = torch.exp(z)
    y_tot = y.sum(dim=1).unsqueeze(1)
    return y / y_tot


# Define cross-entropy loss
def cross_entropy_loss(predicted_probs: torch.tensor, y: torch.tensor) -> torch.tensor:
    # Get a tensor of predicted probabilities for each target class in the mini-batch
    target_probs = predicted_probs[torch.arange(predicted_probs.size(0)), y]

    # Calculate the loss for each example
    loss = -1 * torch.log(target_probs)

    # Return average loss over the examples
    return loss.mean()


# Define a function to save models
def save_model(
    model: nn.Module,
    name: str,
    root_path: str = root_path,
) -> str:
    # Create a directory for models if it doesn't yet exist
    if not os.path.exists(os.path.join(root_path, "models")):
        os.mkdir(os.path.join(root_path, "models"))

    # Save the model to the directory
    filepath = os.path.join(root_path, "models", f"{name}.pt")
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model saved to: {filepath}")

    # Return the filepath
    return filepath


# Define a function for calculating the L1 norm of the gradients
def calculate_l1_norm(named_params):
    norms = []
    for name, param in named_params:
        grad_norm = torch.sum(torch.abs(param.grad)).item()
        norms.append(grad_norm)
    norms = torch.tensor(norms)
    l1_norm = torch.sum(torch.abs(norms)).item()
    return l1_norm


# Define a function to evaluate a single epoch
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train: bool = True,
    lr: float = 1e-2,
    alpha: float = 0.0,
    track_norms: bool = False,
) -> tuple[float, float, list]:
    # Make sure we're on the correct device
    model = model.to(device)

    # Set the model mode – either training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # Initial values for tracking loss and accuracy over the epoch
    running_loss, n_correct, n_total = 0, 0, 0

    # Empty list for tracking gradient size over the epoch
    l1_norms = []

    # Set context based on model mode
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        # Iterate through the batches in the loader
        for idx, (xb, yb) in enumerate(loader):
            # Transfer to device
            xb, yb = xb.to(device), yb.to(device)

            # -- Forward pass -- #
            # Get probabilities for each class
            predicted_probs = model(xb)

            # Get average loss over all the examples in the mini-batch
            loss = cross_entropy_loss(predicted_probs, yb)

            # -- Backward pass -- #
            if train:
                model.backward(loss, lr, alpha)
                if track_norms:
                    l1_norms.append(calculate_l1_norm(model.named_parameters()))

            # Update running counts for loss and accuracy
            running_loss += loss.item()
            predictions = predicted_probs.argmax(dim=1)
            n_correct += (predictions == yb).sum()
            n_total += yb.shape[0]

    # Calculate accuracy for this epoch
    acc = n_correct / n_total

    # Calculate average batch loss
    running_loss = running_loss / len(loader)

    # Return loss and accuracy for this epoch
    return running_loss, acc.item(), l1_norms


# Define function for training
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-2,
    n_epochs: int = 1500,
    print_every: int = 10,
    early_stopping: bool = True,
    patience: int = 100,
    min_delta: float = 1e-2,
    alpha: float = 0.0,
    wandb_config: dict = {},
    wandb_tags: list[str] = [],
    wandb_notes: str = "",
    model_name: str = "baseline",
    track_norms: bool = False,
) -> None:
    # Early stopping setup
    best_test_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch = 1
    no_improvement_count = 0

    # W&B setup
    wandb_config["learning_rate"] = lr
    wandb_config["epochs"] = n_epochs
    wandb_config["batch_size"] = train_loader.batch_size
    wandb_config["early_stopping"] = early_stopping
    wandb_config["patience"] = patience
    wandb_config["min_delta"] = min_delta
    wandb_config["alpha"] = alpha

    # Start training
    with wandb.init(
        entity=entity,
        project=project_name,
        notes=wandb_notes,
        tags=wandb_tags,
        config=wandb_config,
    ) as run:
        for epoch in range(n_epochs):
            # Training over epoch
            if track_norms and epoch == 0:
                train_loss, train_acc, grad_norms = run_epoch(
                    model=model,
                    loader=train_loader,
                    device=device,
                    train=True,
                    lr=lr,
                    alpha=alpha,
                    track_norms=True,
                )
                norms_for_wandb = [
                    [i, grad_norm] for i, grad_norm in enumerate(grad_norms)
                ]
            else:
                train_loss, train_acc, _ = run_epoch(
                    model=model,
                    loader=train_loader,
                    device=device,
                    train=True,
                    lr=lr,
                    alpha=alpha,
                )
                norms_for_wandb = [[i, torch.nan] for i in range(len(train_loader))]

            # Testing over epoch
            test_loss, test_acc, _ = run_epoch(
                model=model,
                loader=test_loader,
                device=device,
                train=False,
                lr=lr,
                alpha=alpha,
            )

            # Save losses and accuracies for this epoch
            table = wandb.Table(
                columns=["Batch", "Gradient L1-norm"], data=norms_for_wandb
            )
            run.log(
                {
                    "training_loss": train_loss,
                    "testing_loss": test_loss,
                    "training_accuracy": train_acc,
                    "testing_accuracy": test_acc,
                    "grad_norms": table,
                }
            )

            # Print the losses periodically
            if epoch % print_every == (print_every - 1):
                logging.info(
                    f"Epoch {epoch+1}/{n_epochs}\n"
                    f"--------------------------\n"
                    f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}\n"
                    f"Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}\n"
                )

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
                        logging.info(
                            f"Early stopping triggered at epoch {epoch+1}\n"
                            f"Best epoch was {best_epoch} with test loss {best_test_loss:.4f}"
                        )
                        break

        # Reload best model
        if early_stopping:
            model.load_state_dict(best_model_weights)

        # Upload the best model as an artifact
        model_filepath = save_model(model=model, name=model_name, root_path=root_path)
        run.log_artifact(model_filepath, name="trained-model", type="model")

    return


# Define a simple two-layer network
class TwoLayerNetwork(nn.Module):

    def __init__(
        self, input_size: int = 3072, hidden_size: int = 512, n_classes: int = 100
    ):
        super(TwoLayerNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.first_pass = True

        # Define all layers in the model
        # layer 1
        self.linear1 = nn.Linear(self.input_size, hidden_size)
        # layer 2
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # layer 3
        self.linear3 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.tensor):
        # Build the feed forward structure
        x = x.flatten(1)
        linear1 = self.linear1(x)
        act1 = sigmoid(linear1)
        linear2 = self.linear2(act1)
        act2 = sigmoid(linear2)
        linear3 = self.linear3(act2)
        output = softmax(linear3)
        return output

    def backward(self, loss: torch.tensor, lr: float, alpha: float = 0.0):
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
        with torch.no_grad():
            self.linear1.weight -= lr * self.linear1.weight.grad
            self.linear1.bias -= lr * self.linear1.bias.grad
            self.linear2.weight -= lr * self.linear2.weight.grad
            self.linear2.bias -= lr * self.linear2.bias.grad
            self.linear3.weight -= lr * self.linear3.weight.grad
            self.linear3.bias -= lr * self.linear3.bias.grad


# Define a baseline network for deep learning
class BaselineDeepNetwork(nn.Module):

    def __init__(
        self,
        input_size: int = 3072,
        n_classes: int = 100,
        activation_function: typing.Callable = sigmoid,
    ):
        super(BaselineDeepNetwork, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.activation_fn = activation_function

        # -- Layer Definitions -- #
        # layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=0
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 3
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # layer 4
        # After first three layers we're left with [N_examples, 64, 2, 2]
        self.linear4 = nn.Linear(64 * 2 * 2, 128)

        # layer 5
        self.linear5 = nn.Linear(128, n_classes)

        # TODO: Fill these in
        # Storage tensors for momentum
        self.conv1_weight_momentum = torch.zeros_like(self.conv1.weight)
        self.conv1_bias_momentum = torch.zeros_like(self.conv1.bias)
        self.conv2_weight_momentum = torch.zeros_like(self.conv2.weight)
        self.conv2_bias_momentum = torch.zeros_like(self.conv2.bias)
        self.conv3_weight_momentum = torch.zeros_like(self.conv3.weight)
        self.conv3_bias_momentum = torch.zeros_like(self.conv3.bias)
        self.linear4_weight_momentum = torch.zeros_like(self.linear4.weight)
        self.linear4_bias_momentum = torch.zeros_like(self.linear4.bias)
        self.linear5_weight_momentum = torch.zeros_like(self.linear5.weight)
        self.linear5_bias_momentum = torch.zeros_like(self.linear5.bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Layer 1
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.pool3(x)

        # Flatten between convolutional and fully connected layers
        x = x.flatten(1)

        # Layer 4
        x = self.linear4(x)
        x = self.activation_fn(x)

        # Layer 5
        x = self.linear5(x)

        # Calculate probabilities from logits
        output = softmax(x)

        return output

    def backward(
        self, loss: torch.tensor, lr: float = 0.01, alpha: float = 0.0
    ) -> None:
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
        with torch.no_grad():
            # TODO: Update momentum
            self.conv1_weight_momentum = (
                alpha * self.conv1_weight_momentum + self.conv1.weight.grad
            )
            self.conv1_bias_momentum = (
                alpha * self.conv1_bias_momentum + self.conv1.bias.grad
            )
            self.conv2_weight_momentum = (
                alpha * self.conv2_weight_momentum + self.conv2.weight.grad
            )
            self.conv2_bias_momentum = (
                alpha * self.conv2_bias_momentum + self.conv2.bias.grad
            )
            self.conv3_weight_momentum = (
                alpha * self.conv3_weight_momentum + self.conv3.weight.grad
            )
            self.conv3_bias_momentum = (
                alpha * self.conv3_bias_momentum + self.conv3.bias.grad
            )
            self.linear4_weight_momentum = (
                alpha * self.linear4_weight_momentum + self.linear4.weight.grad
            )
            self.linear4_bias_momentum = (
                alpha * self.linear4_bias_momentum + self.linear4.bias.grad
            )
            self.linear5_weight_momentum = (
                alpha * self.linear5_weight_momentum + self.linear5.weight.grad
            )
            self.linear5_bias_momentum = (
                alpha * self.linear5_bias_momentum + self.linear5.bias.grad
            )

            self.conv1.weight -= lr * self.conv1_weight_momentum
            self.conv1.bias -= lr * self.conv1_bias_momentum
            self.conv2.weight -= lr * self.conv2_weight_momentum
            self.conv2.bias -= lr * self.conv2_bias_momentum
            self.conv3.weight -= lr * self.conv3_weight_momentum
            self.conv3.bias -= lr * self.conv3_bias_momentum
            self.linear4.weight -= lr * self.linear4_weight_momentum
            self.linear4.bias -= lr * self.linear4_bias_momentum
            self.linear5.weight -= lr * self.linear5_weight_momentum
            self.linear5.bias -= lr * self.linear5_bias_momentum


# Define an extended model for deep learning with skip connections
class ExtendedDeepModel(nn.Module):
    def __init__(
        self,
        input_size: int = 3072,
        n_classes: int = 100,
        activation_function: typing.Callable = sigmoid,
    ):
        super(ExtendedDeepModel, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.activation_fn = activation_function

        # -- Layer Definitions -- #
        # layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=0
        )
        self.conv2_1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 3
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # layer 4
        # After first three layers we're left with [N_examples, 64, 2, 2]
        self.linear4 = nn.Linear(64 * 2 * 2, 128)
        self.linear4_1 = nn.Linear(128, 128)
        self.linear4_2 = nn.Linear(128, 128)
        self.linear4_3 = nn.Linear(128, 128)
        self.linear4_4 = nn.Linear(128, 128)
        self.linear4_5 = nn.Linear(128, 128)

        # layer 5
        self.linear5 = nn.Linear(128, n_classes)

        # Storage tensors for momentum
        self.conv1_weight_momentum = torch.zeros_like(self.conv1.weight).to(
            device=device
        )
        self.conv1_bias_momentum = torch.zeros_like(self.conv1.bias).to(device=device)
        self.conv2_weight_momentum = torch.zeros_like(self.conv2.weight).to(
            device=device
        )
        self.conv2_bias_momentum = torch.zeros_like(self.conv2.bias).to(device=device)

        self.conv2_1_weight_momentum = torch.zeros_like(self.conv2_1.weight).to(
            device=device
        )
        self.conv2_1_bias_momentum = torch.zeros_like(self.conv2_1.bias).to(
            device=device
        )
        self.conv2_2_weight_momentum = torch.zeros_like(self.conv2_2.weight).to(
            device=device
        )
        self.conv2_2_bias_momentum = torch.zeros_like(self.conv2_2.bias).to(
            device=device
        )
        self.conv2_3_weight_momentum = torch.zeros_like(self.conv2_3.weight).to(
            device=device
        )
        self.conv2_3_bias_momentum = torch.zeros_like(self.conv2_3.bias).to(
            device=device
        )
        self.conv2_4_weight_momentum = torch.zeros_like(self.conv2_4.weight).to(
            device=device
        )
        self.conv2_4_bias_momentum = torch.zeros_like(self.conv2_4.bias).to(
            device=device
        )
        self.conv2_5_weight_momentum = torch.zeros_like(self.conv2_5.weight).to(
            device=device
        )
        self.conv2_5_bias_momentum = torch.zeros_like(self.conv2_5.bias).to(
            device=device
        )
        self.conv3_weight_momentum = torch.zeros_like(self.conv3.weight).to(
            device=device
        )
        self.conv3_bias_momentum = torch.zeros_like(self.conv3.bias).to(device=device)
        self.linear4_weight_momentum = torch.zeros_like(self.linear4.weight).to(
            device=device
        )
        self.linear4_bias_momentum = torch.zeros_like(self.linear4.bias).to(
            device=device
        )
        self.linear4_1_weight_momentum = torch.zeros_like(self.linear4_1.weight).to(
            device=device
        )
        self.linear4_1_bias_momentum = torch.zeros_like(self.linear4_1.bias).to(
            device=device
        )
        self.linear4_2_weight_momentum = torch.zeros_like(self.linear4_2.weight).to(
            device=device
        )
        self.linear4_2_bias_momentum = torch.zeros_like(self.linear4_2.bias).to(
            device=device
        )
        self.linear4_3_weight_momentum = torch.zeros_like(self.linear4_3.weight).to(
            device=device
        )
        self.linear4_3_bias_momentum = torch.zeros_like(self.linear4_3.bias).to(
            device=device
        )
        self.linear4_4_weight_momentum = torch.zeros_like(self.linear4_4.weight).to(
            device=device
        )
        self.linear4_4_bias_momentum = torch.zeros_like(self.linear4_4.bias).to(
            device=device
        )
        self.linear4_5_weight_momentum = torch.zeros_like(self.linear4_5.weight).to(
            device=device
        )
        self.linear4_5_bias_momentum = torch.zeros_like(self.linear4_5.bias).to(
            device=device
        )
        self.linear5_weight_momentum = torch.zeros_like(self.linear5.weight).to(
            device=device
        )
        self.linear5_bias_momentum = torch.zeros_like(self.linear5.bias).to(
            device=device
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Layer 1
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.activation_fn(x)

        # Layer 3
        x = self.conv2_1(x)
        x = self.activation_fn(x)

        # Layer 4
        x = self.conv2_2(x)
        x = self.activation_fn(x)

        # Layer 5
        x = self.conv2_3(x)
        x = self.activation_fn(x)

        # Layer 6
        x = self.conv2_4(x)
        x = self.activation_fn(x)

        # Layer 7
        x = self.conv2_5(x)
        x = self.activation_fn(x)
        x = self.pool2(x)

        # Layer 8
        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.pool3(x)

        # Flatten between convolutional and fully connected layers
        x = x.flatten(1)

        # Layer 9
        x = self.linear4(x)
        x = self.activation_fn(x)

        # Layer 10
        x = self.linear4_1(x)
        x = self.activation_fn(x)

        # Layer 11
        x = self.linear4_2(x)
        x = self.activation_fn(x)

        # Layer 12
        x = self.linear4_3(x)
        x = self.activation_fn(x)

        # Layer 13
        x = self.linear4_4(x)
        x = self.activation_fn(x)

        # Layer 14
        x = self.linear4_5(x)
        x = self.activation_fn(x)

        # Layer 15
        x = self.linear5(x)

        # Calculate probabilities from logits
        output = softmax(x)

        return output

    def backward(
        self, loss: torch.tensor, lr: float = 0.01, alpha: float = 0.0
    ) -> None:
        # Reset parameter gradients
        # NOTE: No learnable parameters for activation functions or pooling layers
        self.conv1.weight.grad = None
        self.conv1.bias.grad = None
        self.conv2.weight.grad = None
        self.conv2.bias.grad = None
        self.conv2_1.weight.grad = None
        self.conv2_1.bias.grad = None
        self.conv2_2.weight.grad = None
        self.conv2_2.bias.grad = None
        self.conv2_3.weight.grad = None
        self.conv2_3.bias.grad = None
        self.conv2_4.weight.grad = None
        self.conv2_4.bias.grad = None
        self.conv2_5.weight.grad = None
        self.conv2_5.bias.grad = None
        self.conv3.weight.grad = None
        self.conv3.bias.grad = None
        self.linear4.weight.grad = None
        self.linear4.bias.grad = None
        self.linear4_1.weight.grad = None
        self.linear4_1.bias.grad = None
        self.linear4_2.weight.grad = None
        self.linear4_2.bias.grad = None
        self.linear4_3.weight.grad = None
        self.linear4_3.bias.grad = None
        self.linear4_4.weight.grad = None
        self.linear4_4.bias.grad = None
        self.linear4_5.weight.grad = None
        self.linear4_5.bias.grad = None
        self.linear5.weight.grad = None
        self.linear5.bias.grad = None

        # Update gradients
        loss.backward()

        # Update parameters
        with torch.no_grad():
            # Update momentum
            self.conv1_weight_momentum = (
                alpha * self.conv1_weight_momentum + self.conv1.weight.grad
            )
            self.conv1_bias_momentum = (
                alpha * self.conv1_bias_momentum + self.conv1.bias.grad
            )
            self.conv2_weight_momentum = (
                alpha * self.conv2_weight_momentum + self.conv2.weight.grad
            )
            self.conv2_bias_momentum = (
                alpha * self.conv2_bias_momentum + self.conv2.bias.grad
            )
            self.conv2_1_weight_momentum = (
                alpha * self.conv2_1_weight_momentum + self.conv2_1.weight.grad
            )
            self.conv2_1_bias_momentum = (
                alpha * self.conv2_1_bias_momentum + self.conv2_1.bias.grad
            )
            self.conv2_2_weight_momentum = (
                alpha * self.conv2_2_weight_momentum + self.conv2_2.weight.grad
            )
            self.conv2_2_bias_momentum = (
                alpha * self.conv2_2_bias_momentum + self.conv2_2.bias.grad
            )
            self.conv2_3_weight_momentum = (
                alpha * self.conv2_3_weight_momentum + self.conv2_3.weight.grad
            )
            self.conv2_3_bias_momentum = (
                alpha * self.conv2_3_bias_momentum + self.conv2_3.bias.grad
            )
            self.conv2_4_weight_momentum = (
                alpha * self.conv2_4_weight_momentum + self.conv2_4.weight.grad
            )
            self.conv2_4_bias_momentum = (
                alpha * self.conv2_4_bias_momentum + self.conv2_4.bias.grad
            )
            self.conv2_5_weight_momentum = (
                alpha * self.conv2_5_weight_momentum + self.conv2_5.weight.grad
            )
            self.conv2_5_bias_momentum = (
                alpha * self.conv2_5_bias_momentum + self.conv2_5.bias.grad
            )
            self.conv3_weight_momentum = (
                alpha * self.conv3_weight_momentum + self.conv3.weight.grad
            )
            self.conv3_bias_momentum = (
                alpha * self.conv3_bias_momentum + self.conv3.bias.grad
            )
            self.linear4_weight_momentum = (
                alpha * self.linear4_weight_momentum + self.linear4.weight.grad
            )
            self.linear4_bias_momentum = (
                alpha * self.linear4_bias_momentum + self.linear4.bias.grad
            )
            self.linear4_1_weight_momentum = (
                alpha * self.linear4_1_weight_momentum + self.linear4_1.weight.grad
            )
            self.linear4_1_bias_momentum = (
                alpha * self.linear4_1_bias_momentum + self.linear4_1.bias.grad
            )
            self.linear4_2_weight_momentum = (
                alpha * self.linear4_2_weight_momentum + self.linear4_2.weight.grad
            )
            self.linear4_2_bias_momentum = (
                alpha * self.linear4_2_bias_momentum + self.linear4_2.bias.grad
            )
            self.linear4_3_weight_momentum = (
                alpha * self.linear4_3_weight_momentum + self.linear4_3.weight.grad
            )
            self.linear4_3_bias_momentum = (
                alpha * self.linear4_3_bias_momentum + self.linear4_3.bias.grad
            )
            self.linear4_4_weight_momentum = (
                alpha * self.linear4_4_weight_momentum + self.linear4_4.weight.grad
            )
            self.linear4_4_bias_momentum = (
                alpha * self.linear4_4_bias_momentum + self.linear4_4.bias.grad
            )
            self.linear4_5_weight_momentum = (
                alpha * self.linear4_5_weight_momentum + self.linear4_5.weight.grad
            )
            self.linear4_5_bias_momentum = (
                alpha * self.linear4_5_bias_momentum + self.linear4_5.bias.grad
            )
            self.linear5_weight_momentum = (
                alpha * self.linear5_weight_momentum + self.linear5.weight.grad
            )
            self.linear5_bias_momentum = (
                alpha * self.linear5_bias_momentum + self.linear5.bias.grad
            )

            self.conv1.weight -= lr * self.conv1_weight_momentum
            self.conv1.bias -= lr * self.conv1_bias_momentum
            self.conv2.weight -= lr * self.conv2_weight_momentum
            self.conv2.bias -= lr * self.conv2_bias_momentum
            self.conv2_1.weight -= lr * self.conv2_1_weight_momentum
            self.conv2_1.bias -= lr * self.conv2_1_bias_momentum
            self.conv2_2.weight -= lr * self.conv2_2_weight_momentum
            self.conv2_2.bias -= lr * self.conv2_2_bias_momentum
            self.conv2_3.weight -= lr * self.conv2_3_weight_momentum
            self.conv2_3.bias -= lr * self.conv2_3_bias_momentum
            self.conv2_4.weight -= lr * self.conv2_4_weight_momentum
            self.conv2_4.bias -= lr * self.conv2_4_bias_momentum
            self.conv2_5.weight -= lr * self.conv2_5_weight_momentum
            self.conv2_5.bias -= lr * self.conv2_5_bias_momentum
            self.conv3.weight -= lr * self.conv3_weight_momentum
            self.conv3.bias -= lr * self.conv3_bias_momentum
            self.linear4.weight -= lr * self.linear4_weight_momentum
            self.linear4.bias -= lr * self.linear4_bias_momentum
            self.linear4_1.weight -= lr * self.linear4_1_weight_momentum
            self.linear4_1.bias -= lr * self.linear4_1_bias_momentum
            self.linear4_2.weight -= lr * self.linear4_2_weight_momentum
            self.linear4_2.bias -= lr * self.linear4_2_bias_momentum
            self.linear4_3.weight -= lr * self.linear4_3_weight_momentum
            self.linear4_3.bias -= lr * self.linear4_3_bias_momentum
            self.linear4_4.weight -= lr * self.linear4_4_weight_momentum
            self.linear4_4.bias -= lr * self.linear4_4_bias_momentum
            self.linear4_5.weight -= lr * self.linear4_5_weight_momentum
            self.linear4_5.bias -= lr * self.linear4_5_bias_momentum
            self.linear5.weight -= lr * self.linear5_weight_momentum
            self.linear5.bias -= lr * self.linear5_bias_momentum


class SkipConnectionModel1(nn.Module):
    def __init__(
        self,
        input_size: int = 3072,
        n_classes: int = 100,
        activation_function: typing.Callable = sigmoid,
    ):
        super(SkipConnectionModel1, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.activation_fn = activation_function

        # -- Layer Definitions -- #
        # layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=0
        )
        self.conv2_1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 3
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # layer 4
        # After first three layers we're left with [N_examples, 64, 2, 2]
        self.linear4 = nn.Linear(64 * 2 * 2, 128)
        self.linear4_1 = nn.Linear(128, 128)
        self.linear4_2 = nn.Linear(128, 128)
        self.linear4_3 = nn.Linear(128, 128)
        self.linear4_4 = nn.Linear(128, 128)
        self.linear4_5 = nn.Linear(128, 128)

        # layer 5
        self.linear5 = nn.Linear(128, n_classes)

        # Storage tensors for momentum
        self.conv1_weight_momentum = torch.zeros_like(self.conv1.weight).to(
            device=device
        )
        self.conv1_bias_momentum = torch.zeros_like(self.conv1.bias).to(device=device)
        self.conv2_weight_momentum = torch.zeros_like(self.conv2.weight).to(
            device=device
        )
        self.conv2_bias_momentum = torch.zeros_like(self.conv2.bias).to(device=device)

        self.conv2_1_weight_momentum = torch.zeros_like(self.conv2_1.weight).to(
            device=device
        )
        self.conv2_1_bias_momentum = torch.zeros_like(self.conv2_1.bias).to(
            device=device
        )
        self.conv2_2_weight_momentum = torch.zeros_like(self.conv2_2.weight).to(
            device=device
        )
        self.conv2_2_bias_momentum = torch.zeros_like(self.conv2_2.bias).to(
            device=device
        )
        self.conv2_3_weight_momentum = torch.zeros_like(self.conv2_3.weight).to(
            device=device
        )
        self.conv2_3_bias_momentum = torch.zeros_like(self.conv2_3.bias).to(
            device=device
        )
        self.conv2_4_weight_momentum = torch.zeros_like(self.conv2_4.weight).to(
            device=device
        )
        self.conv2_4_bias_momentum = torch.zeros_like(self.conv2_4.bias).to(
            device=device
        )
        self.conv2_5_weight_momentum = torch.zeros_like(self.conv2_5.weight).to(
            device=device
        )
        self.conv2_5_bias_momentum = torch.zeros_like(self.conv2_5.bias).to(
            device=device
        )
        self.conv3_weight_momentum = torch.zeros_like(self.conv3.weight).to(
            device=device
        )
        self.conv3_bias_momentum = torch.zeros_like(self.conv3.bias).to(device=device)
        self.linear4_weight_momentum = torch.zeros_like(self.linear4.weight).to(
            device=device
        )
        self.linear4_bias_momentum = torch.zeros_like(self.linear4.bias).to(
            device=device
        )
        self.linear4_1_weight_momentum = torch.zeros_like(self.linear4_1.weight).to(
            device=device
        )
        self.linear4_1_bias_momentum = torch.zeros_like(self.linear4_1.bias).to(
            device=device
        )
        self.linear4_2_weight_momentum = torch.zeros_like(self.linear4_2.weight).to(
            device=device
        )
        self.linear4_2_bias_momentum = torch.zeros_like(self.linear4_2.bias).to(
            device=device
        )
        self.linear4_3_weight_momentum = torch.zeros_like(self.linear4_3.weight).to(
            device=device
        )
        self.linear4_3_bias_momentum = torch.zeros_like(self.linear4_3.bias).to(
            device=device
        )
        self.linear4_4_weight_momentum = torch.zeros_like(self.linear4_4.weight).to(
            device=device
        )
        self.linear4_4_bias_momentum = torch.zeros_like(self.linear4_4.bias).to(
            device=device
        )
        self.linear4_5_weight_momentum = torch.zeros_like(self.linear4_5.weight).to(
            device=device
        )
        self.linear4_5_bias_momentum = torch.zeros_like(self.linear4_5.bias).to(
            device=device
        )
        self.linear5_weight_momentum = torch.zeros_like(self.linear5.weight).to(
            device=device
        )
        self.linear5_bias_momentum = torch.zeros_like(self.linear5.bias).to(
            device=device
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Layer 1
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x_1 = self.activation_fn(x)  # rename for skip connection

        # Layer 3
        x_2 = self.conv2_1(x_1) + x  # Skip connection #1
        x = self.activation_fn(x_2)

        # Layer 4
        x_1 = self.conv2_2(x)  # rename for skip connection
        x_2 = self.activation_fn(x_1)

        # Layer 5
        x_3 = self.conv2_3(x_2)
        x_4 = self.activation_fn(x_3) + x  # Skip connection #2

        # Layer 6
        x = self.conv2_4(x_4)
        x = self.activation_fn(x)

        # Layer 7
        x = self.conv2_5(x)
        x = self.activation_fn(x)
        x = self.pool2(x)

        # Layer 8
        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.pool3(x)

        # Flatten between convolutional and fully connected layers
        x = x.flatten(1)

        # Layer 9
        x = self.linear4(x)
        x_1 = self.activation_fn(x)  # rename for skip connection

        # Layer 10
        x_2 = self.linear4_1(x_1) + x  # Skip connection #3
        x = self.activation_fn(x_2)

        # Layer 11
        x = self.linear4_2(x)
        x = self.activation_fn(x)

        # Layer 12
        x = self.linear4_3(x)
        x = self.activation_fn(x)

        # Layer 13
        x = self.linear4_4(x)
        x = self.activation_fn(x)

        # Layer 14
        x = self.linear4_5(x)
        x = self.activation_fn(x)

        # Layer 15
        x = self.linear5(x)

        # Calculate probabilities from logits
        output = softmax(x)

        return output

    def backward(
        self, loss: torch.tensor, lr: float = 0.01, alpha: float = 0.0
    ) -> None:
        # Reset parameter gradients
        # NOTE: No learnable parameters for activation functions or pooling layers
        self.conv1.weight.grad = None
        self.conv1.bias.grad = None
        self.conv2.weight.grad = None
        self.conv2.bias.grad = None
        self.conv2_1.weight.grad = None
        self.conv2_1.bias.grad = None
        self.conv2_2.weight.grad = None
        self.conv2_2.bias.grad = None
        self.conv2_3.weight.grad = None
        self.conv2_3.bias.grad = None
        self.conv2_4.weight.grad = None
        self.conv2_4.bias.grad = None
        self.conv2_5.weight.grad = None
        self.conv2_5.bias.grad = None
        self.conv3.weight.grad = None
        self.conv3.bias.grad = None
        self.linear4.weight.grad = None
        self.linear4.bias.grad = None
        self.linear4_1.weight.grad = None
        self.linear4_1.bias.grad = None
        self.linear4_2.weight.grad = None
        self.linear4_2.bias.grad = None
        self.linear4_3.weight.grad = None
        self.linear4_3.bias.grad = None
        self.linear4_4.weight.grad = None
        self.linear4_4.bias.grad = None
        self.linear4_5.weight.grad = None
        self.linear4_5.bias.grad = None
        self.linear5.weight.grad = None
        self.linear5.bias.grad = None

        # Update gradients
        loss.backward()

        # Update parameters
        with torch.no_grad():
            # Update momentum
            self.conv1_weight_momentum = (
                alpha * self.conv1_weight_momentum + self.conv1.weight.grad
            )
            self.conv1_bias_momentum = (
                alpha * self.conv1_bias_momentum + self.conv1.bias.grad
            )
            self.conv2_weight_momentum = (
                alpha * self.conv2_weight_momentum + self.conv2.weight.grad
            )
            self.conv2_bias_momentum = (
                alpha * self.conv2_bias_momentum + self.conv2.bias.grad
            )
            self.conv2_1_weight_momentum = (
                alpha * self.conv2_1_weight_momentum + self.conv2_1.weight.grad
            )
            self.conv2_1_bias_momentum = (
                alpha * self.conv2_1_bias_momentum + self.conv2_1.bias.grad
            )
            self.conv2_2_weight_momentum = (
                alpha * self.conv2_2_weight_momentum + self.conv2_2.weight.grad
            )
            self.conv2_2_bias_momentum = (
                alpha * self.conv2_2_bias_momentum + self.conv2_2.bias.grad
            )
            self.conv2_3_weight_momentum = (
                alpha * self.conv2_3_weight_momentum + self.conv2_3.weight.grad
            )
            self.conv2_3_bias_momentum = (
                alpha * self.conv2_3_bias_momentum + self.conv2_3.bias.grad
            )
            self.conv2_4_weight_momentum = (
                alpha * self.conv2_4_weight_momentum + self.conv2_4.weight.grad
            )
            self.conv2_4_bias_momentum = (
                alpha * self.conv2_4_bias_momentum + self.conv2_4.bias.grad
            )
            self.conv2_5_weight_momentum = (
                alpha * self.conv2_5_weight_momentum + self.conv2_5.weight.grad
            )
            self.conv2_5_bias_momentum = (
                alpha * self.conv2_5_bias_momentum + self.conv2_5.bias.grad
            )
            self.conv3_weight_momentum = (
                alpha * self.conv3_weight_momentum + self.conv3.weight.grad
            )
            self.conv3_bias_momentum = (
                alpha * self.conv3_bias_momentum + self.conv3.bias.grad
            )
            self.linear4_weight_momentum = (
                alpha * self.linear4_weight_momentum + self.linear4.weight.grad
            )
            self.linear4_bias_momentum = (
                alpha * self.linear4_bias_momentum + self.linear4.bias.grad
            )
            self.linear4_1_weight_momentum = (
                alpha * self.linear4_1_weight_momentum + self.linear4_1.weight.grad
            )
            self.linear4_1_bias_momentum = (
                alpha * self.linear4_1_bias_momentum + self.linear4_1.bias.grad
            )
            self.linear4_2_weight_momentum = (
                alpha * self.linear4_2_weight_momentum + self.linear4_2.weight.grad
            )
            self.linear4_2_bias_momentum = (
                alpha * self.linear4_2_bias_momentum + self.linear4_2.bias.grad
            )
            self.linear4_3_weight_momentum = (
                alpha * self.linear4_3_weight_momentum + self.linear4_3.weight.grad
            )
            self.linear4_3_bias_momentum = (
                alpha * self.linear4_3_bias_momentum + self.linear4_3.bias.grad
            )
            self.linear4_4_weight_momentum = (
                alpha * self.linear4_4_weight_momentum + self.linear4_4.weight.grad
            )
            self.linear4_4_bias_momentum = (
                alpha * self.linear4_4_bias_momentum + self.linear4_4.bias.grad
            )
            self.linear4_5_weight_momentum = (
                alpha * self.linear4_5_weight_momentum + self.linear4_5.weight.grad
            )
            self.linear4_5_bias_momentum = (
                alpha * self.linear4_5_bias_momentum + self.linear4_5.bias.grad
            )
            self.linear5_weight_momentum = (
                alpha * self.linear5_weight_momentum + self.linear5.weight.grad
            )
            self.linear5_bias_momentum = (
                alpha * self.linear5_bias_momentum + self.linear5.bias.grad
            )

            self.conv1.weight -= lr * self.conv1_weight_momentum
            self.conv1.bias -= lr * self.conv1_bias_momentum
            self.conv2.weight -= lr * self.conv2_weight_momentum
            self.conv2.bias -= lr * self.conv2_bias_momentum
            self.conv2_1.weight -= lr * self.conv2_1_weight_momentum
            self.conv2_1.bias -= lr * self.conv2_1_bias_momentum
            self.conv2_2.weight -= lr * self.conv2_2_weight_momentum
            self.conv2_2.bias -= lr * self.conv2_2_bias_momentum
            self.conv2_3.weight -= lr * self.conv2_3_weight_momentum
            self.conv2_3.bias -= lr * self.conv2_3_bias_momentum
            self.conv2_4.weight -= lr * self.conv2_4_weight_momentum
            self.conv2_4.bias -= lr * self.conv2_4_bias_momentum
            self.conv2_5.weight -= lr * self.conv2_5_weight_momentum
            self.conv2_5.bias -= lr * self.conv2_5_bias_momentum
            self.conv3.weight -= lr * self.conv3_weight_momentum
            self.conv3.bias -= lr * self.conv3_bias_momentum
            self.linear4.weight -= lr * self.linear4_weight_momentum
            self.linear4.bias -= lr * self.linear4_bias_momentum
            self.linear4_1.weight -= lr * self.linear4_1_weight_momentum
            self.linear4_1.bias -= lr * self.linear4_1_bias_momentum
            self.linear4_2.weight -= lr * self.linear4_2_weight_momentum
            self.linear4_2.bias -= lr * self.linear4_2_bias_momentum
            self.linear4_3.weight -= lr * self.linear4_3_weight_momentum
            self.linear4_3.bias -= lr * self.linear4_3_bias_momentum
            self.linear4_4.weight -= lr * self.linear4_4_weight_momentum
            self.linear4_4.bias -= lr * self.linear4_4_bias_momentum
            self.linear4_5.weight -= lr * self.linear4_5_weight_momentum
            self.linear4_5.bias -= lr * self.linear4_5_bias_momentum
            self.linear5.weight -= lr * self.linear5_weight_momentum
            self.linear5.bias -= lr * self.linear5_bias_momentum


class SkipConnectionModel2(nn.Module):
    def __init__(
        self,
        input_size: int = 3072,
        n_classes: int = 100,
        activation_function: typing.Callable = sigmoid,
    ):
        super(SkipConnectionModel2, self).__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.activation_fn = activation_function

        # -- Layer Definitions -- #
        # layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=0
        )
        self.conv2_1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv2_5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        # layer 3
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # layer 4
        # After first three layers we're left with [N_examples, 64, 2, 2]
        self.linear4 = nn.Linear(64 * 2 * 2, 128)
        self.linear4_1 = nn.Linear(128, 128)
        self.linear4_2 = nn.Linear(128, 128)
        self.linear4_3 = nn.Linear(128, 128)
        self.linear4_4 = nn.Linear(128, 128)
        self.linear4_5 = nn.Linear(128, 128)

        # layer 5
        self.linear5 = nn.Linear(128, n_classes)

        # Storage tensors for momentum
        self.conv1_weight_momentum = torch.zeros_like(self.conv1.weight).to(
            device=device
        )
        self.conv1_bias_momentum = torch.zeros_like(self.conv1.bias).to(device=device)
        self.conv2_weight_momentum = torch.zeros_like(self.conv2.weight).to(
            device=device
        )
        self.conv2_bias_momentum = torch.zeros_like(self.conv2.bias).to(device=device)

        self.conv2_1_weight_momentum = torch.zeros_like(self.conv2_1.weight).to(
            device=device
        )
        self.conv2_1_bias_momentum = torch.zeros_like(self.conv2_1.bias).to(
            device=device
        )
        self.conv2_2_weight_momentum = torch.zeros_like(self.conv2_2.weight).to(
            device=device
        )
        self.conv2_2_bias_momentum = torch.zeros_like(self.conv2_2.bias).to(
            device=device
        )
        self.conv2_3_weight_momentum = torch.zeros_like(self.conv2_3.weight).to(
            device=device
        )
        self.conv2_3_bias_momentum = torch.zeros_like(self.conv2_3.bias).to(
            device=device
        )
        self.conv2_4_weight_momentum = torch.zeros_like(self.conv2_4.weight).to(
            device=device
        )
        self.conv2_4_bias_momentum = torch.zeros_like(self.conv2_4.bias).to(
            device=device
        )
        self.conv2_5_weight_momentum = torch.zeros_like(self.conv2_5.weight).to(
            device=device
        )
        self.conv2_5_bias_momentum = torch.zeros_like(self.conv2_5.bias).to(
            device=device
        )
        self.conv3_weight_momentum = torch.zeros_like(self.conv3.weight).to(
            device=device
        )
        self.conv3_bias_momentum = torch.zeros_like(self.conv3.bias).to(device=device)
        self.linear4_weight_momentum = torch.zeros_like(self.linear4.weight).to(
            device=device
        )
        self.linear4_bias_momentum = torch.zeros_like(self.linear4.bias).to(
            device=device
        )
        self.linear4_1_weight_momentum = torch.zeros_like(self.linear4_1.weight).to(
            device=device
        )
        self.linear4_1_bias_momentum = torch.zeros_like(self.linear4_1.bias).to(
            device=device
        )
        self.linear4_2_weight_momentum = torch.zeros_like(self.linear4_2.weight).to(
            device=device
        )
        self.linear4_2_bias_momentum = torch.zeros_like(self.linear4_2.bias).to(
            device=device
        )
        self.linear4_3_weight_momentum = torch.zeros_like(self.linear4_3.weight).to(
            device=device
        )
        self.linear4_3_bias_momentum = torch.zeros_like(self.linear4_3.bias).to(
            device=device
        )
        self.linear4_4_weight_momentum = torch.zeros_like(self.linear4_4.weight).to(
            device=device
        )
        self.linear4_4_bias_momentum = torch.zeros_like(self.linear4_4.bias).to(
            device=device
        )
        self.linear4_5_weight_momentum = torch.zeros_like(self.linear4_5.weight).to(
            device=device
        )
        self.linear4_5_bias_momentum = torch.zeros_like(self.linear4_5.bias).to(
            device=device
        )
        self.linear5_weight_momentum = torch.zeros_like(self.linear5.weight).to(
            device=device
        )
        self.linear5_bias_momentum = torch.zeros_like(self.linear5.bias).to(
            device=device
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Layer 1
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.pool1(x)

        # Layer 2
        x_1 = self.conv2(x)  # rename for skip connection
        x = self.activation_fn(x_1)

        # Layer 3
        x = self.conv2_1(x)
        x = self.activation_fn(x)

        # Layer 4
        x = self.conv2_2(x)
        x_2 = self.activation_fn(x) + x_1  # Skip connection #1

        # Layer 5
        x = self.conv2_3(x_2)
        x = self.activation_fn(x)

        # Layer 6
        x = self.conv2_4(x)
        x = self.activation_fn(x)

        # Layer 7
        x = self.conv2_5(x)
        x = self.activation_fn(x) + x_2  # Skip connection #2
        x = self.pool2(x)

        # Layer 8
        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.pool3(x)

        # Flatten between convolutional and fully connected layers
        x = x.flatten(1)

        # Layer 9
        x_1 = self.linear4(x)  # rename for skip connection
        x = self.activation_fn(x_1)

        # Layer 10
        x = self.linear4_1(x)
        x = self.activation_fn(x)

        # Layer 11
        x = self.linear4_2(x)
        x = self.activation_fn(x)

        # Layer 12
        x = self.linear4_3(x)
        x = self.activation_fn(x)

        # Layer 13
        x = self.linear4_4(x)
        x = self.activation_fn(x)

        # Layer 14
        x = self.linear4_5(x)
        x = self.activation_fn(x) + x_1  # Skip connection #3

        # Layer 15
        x = self.linear5(x)

        # Calculate probabilities from logits
        output = softmax(x)

        return output

    def backward(
        self, loss: torch.tensor, lr: float = 0.01, alpha: float = 0.0
    ) -> None:
        # Reset parameter gradients
        # NOTE: No learnable parameters for activation functions or pooling layers
        self.conv1.weight.grad = None
        self.conv1.bias.grad = None
        self.conv2.weight.grad = None
        self.conv2.bias.grad = None
        self.conv2_1.weight.grad = None
        self.conv2_1.bias.grad = None
        self.conv2_2.weight.grad = None
        self.conv2_2.bias.grad = None
        self.conv2_3.weight.grad = None
        self.conv2_3.bias.grad = None
        self.conv2_4.weight.grad = None
        self.conv2_4.bias.grad = None
        self.conv2_5.weight.grad = None
        self.conv2_5.bias.grad = None
        self.conv3.weight.grad = None
        self.conv3.bias.grad = None
        self.linear4.weight.grad = None
        self.linear4.bias.grad = None
        self.linear4_1.weight.grad = None
        self.linear4_1.bias.grad = None
        self.linear4_2.weight.grad = None
        self.linear4_2.bias.grad = None
        self.linear4_3.weight.grad = None
        self.linear4_3.bias.grad = None
        self.linear4_4.weight.grad = None
        self.linear4_4.bias.grad = None
        self.linear4_5.weight.grad = None
        self.linear4_5.bias.grad = None
        self.linear5.weight.grad = None
        self.linear5.bias.grad = None

        # Update gradients
        loss.backward()

        # Update parameters
        with torch.no_grad():
            # Update momentum
            self.conv1_weight_momentum = (
                alpha * self.conv1_weight_momentum + self.conv1.weight.grad
            )
            self.conv1_bias_momentum = (
                alpha * self.conv1_bias_momentum + self.conv1.bias.grad
            )
            self.conv2_weight_momentum = (
                alpha * self.conv2_weight_momentum + self.conv2.weight.grad
            )
            self.conv2_bias_momentum = (
                alpha * self.conv2_bias_momentum + self.conv2.bias.grad
            )
            self.conv2_1_weight_momentum = (
                alpha * self.conv2_1_weight_momentum + self.conv2_1.weight.grad
            )
            self.conv2_1_bias_momentum = (
                alpha * self.conv2_1_bias_momentum + self.conv2_1.bias.grad
            )
            self.conv2_2_weight_momentum = (
                alpha * self.conv2_2_weight_momentum + self.conv2_2.weight.grad
            )
            self.conv2_2_bias_momentum = (
                alpha * self.conv2_2_bias_momentum + self.conv2_2.bias.grad
            )
            self.conv2_3_weight_momentum = (
                alpha * self.conv2_3_weight_momentum + self.conv2_3.weight.grad
            )
            self.conv2_3_bias_momentum = (
                alpha * self.conv2_3_bias_momentum + self.conv2_3.bias.grad
            )
            self.conv2_4_weight_momentum = (
                alpha * self.conv2_4_weight_momentum + self.conv2_4.weight.grad
            )
            self.conv2_4_bias_momentum = (
                alpha * self.conv2_4_bias_momentum + self.conv2_4.bias.grad
            )
            self.conv2_5_weight_momentum = (
                alpha * self.conv2_5_weight_momentum + self.conv2_5.weight.grad
            )
            self.conv2_5_bias_momentum = (
                alpha * self.conv2_5_bias_momentum + self.conv2_5.bias.grad
            )
            self.conv3_weight_momentum = (
                alpha * self.conv3_weight_momentum + self.conv3.weight.grad
            )
            self.conv3_bias_momentum = (
                alpha * self.conv3_bias_momentum + self.conv3.bias.grad
            )
            self.linear4_weight_momentum = (
                alpha * self.linear4_weight_momentum + self.linear4.weight.grad
            )
            self.linear4_bias_momentum = (
                alpha * self.linear4_bias_momentum + self.linear4.bias.grad
            )
            self.linear4_1_weight_momentum = (
                alpha * self.linear4_1_weight_momentum + self.linear4_1.weight.grad
            )
            self.linear4_1_bias_momentum = (
                alpha * self.linear4_1_bias_momentum + self.linear4_1.bias.grad
            )
            self.linear4_2_weight_momentum = (
                alpha * self.linear4_2_weight_momentum + self.linear4_2.weight.grad
            )
            self.linear4_2_bias_momentum = (
                alpha * self.linear4_2_bias_momentum + self.linear4_2.bias.grad
            )
            self.linear4_3_weight_momentum = (
                alpha * self.linear4_3_weight_momentum + self.linear4_3.weight.grad
            )
            self.linear4_3_bias_momentum = (
                alpha * self.linear4_3_bias_momentum + self.linear4_3.bias.grad
            )
            self.linear4_4_weight_momentum = (
                alpha * self.linear4_4_weight_momentum + self.linear4_4.weight.grad
            )
            self.linear4_4_bias_momentum = (
                alpha * self.linear4_4_bias_momentum + self.linear4_4.bias.grad
            )
            self.linear4_5_weight_momentum = (
                alpha * self.linear4_5_weight_momentum + self.linear4_5.weight.grad
            )
            self.linear4_5_bias_momentum = (
                alpha * self.linear4_5_bias_momentum + self.linear4_5.bias.grad
            )
            self.linear5_weight_momentum = (
                alpha * self.linear5_weight_momentum + self.linear5.weight.grad
            )
            self.linear5_bias_momentum = (
                alpha * self.linear5_bias_momentum + self.linear5.bias.grad
            )

            self.conv1.weight -= lr * self.conv1_weight_momentum
            self.conv1.bias -= lr * self.conv1_bias_momentum
            self.conv2.weight -= lr * self.conv2_weight_momentum
            self.conv2.bias -= lr * self.conv2_bias_momentum
            self.conv2_1.weight -= lr * self.conv2_1_weight_momentum
            self.conv2_1.bias -= lr * self.conv2_1_bias_momentum
            self.conv2_2.weight -= lr * self.conv2_2_weight_momentum
            self.conv2_2.bias -= lr * self.conv2_2_bias_momentum
            self.conv2_3.weight -= lr * self.conv2_3_weight_momentum
            self.conv2_3.bias -= lr * self.conv2_3_bias_momentum
            self.conv2_4.weight -= lr * self.conv2_4_weight_momentum
            self.conv2_4.bias -= lr * self.conv2_4_bias_momentum
            self.conv2_5.weight -= lr * self.conv2_5_weight_momentum
            self.conv2_5.bias -= lr * self.conv2_5_bias_momentum
            self.conv3.weight -= lr * self.conv3_weight_momentum
            self.conv3.bias -= lr * self.conv3_bias_momentum
            self.linear4.weight -= lr * self.linear4_weight_momentum
            self.linear4.bias -= lr * self.linear4_bias_momentum
            self.linear4_1.weight -= lr * self.linear4_1_weight_momentum
            self.linear4_1.bias -= lr * self.linear4_1_bias_momentum
            self.linear4_2.weight -= lr * self.linear4_2_weight_momentum
            self.linear4_2.bias -= lr * self.linear4_2_bias_momentum
            self.linear4_3.weight -= lr * self.linear4_3_weight_momentum
            self.linear4_3.bias -= lr * self.linear4_3_bias_momentum
            self.linear4_4.weight -= lr * self.linear4_4_weight_momentum
            self.linear4_4.bias -= lr * self.linear4_4_bias_momentum
            self.linear4_5.weight -= lr * self.linear4_5_weight_momentum
            self.linear4_5.bias -= lr * self.linear4_5_bias_momentum
            self.linear5.weight -= lr * self.linear5_weight_momentum
            self.linear5.bias -= lr * self.linear5_bias_momentum


# %% ----- Evaluating the Dataset Difficulty: Training -----
# Create an instance of the model
model = TwoLayerNetwork().to(device)

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    wandb_tags=["baseline", "shallow"],
    model_name="baseline_shallow",
)

# %% ----- Building a Baseline Deep Network: Training -----
# Create an instance of the model
model = BaselineDeepNetwork().to(device)

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    wandb_tags=["baseline", "deep"],
    model_name="baseline_deep",
)

# %% Part 2
########################################
# Activation Functions and Optimizers ##
########################################

# %% ----- Activation Functions: Definitions -----


# Define ReLU activation function
def ReLU(z: torch.tensor) -> torch.tensor:
    return torch.clamp(z, min=0)


# Define Leaky ReLU activation function
def leaky_ReLU(z: torch.tensor) -> torch.tensor:
    return torch.clamp(z, min=0.1 * z)


# Define tanh activation function
def tanh(x: torch.tensor) -> torch.tensor:
    pos_exp = torch.exp(x)
    neg_exp = torch.exp(-x)
    return (pos_exp - neg_exp) / (pos_exp + neg_exp)


# Define SiLU activation function
def SiLU(z: torch.tensor) -> torch.tensor:
    return z * sigmoid(z)


# %% ----- Activation Functions: Training (Pt 1) -----

# Define a modified deep network, replacing the sigmoid activation function with tanh
model = BaselineDeepNetwork(activation_function=tanh).to(device)

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    patience=50,
    wandb_tags=["tanh_activation", "deep", "repeat"],
    model_name="tanh_deep",
)

# %% ----- Activation Functions: Training (Pt 2) -----

# Define a modified deep network, replacing the sigmoid activation function with SiLU
model = BaselineDeepNetwork(activation_function=SiLU).to(device)

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    wandb_tags=["SiLU_activation", "deep", "for_report"],
    model_name="SiLU_deep",
)

# %% ----- Optimizers: Mini-batch SGD -----
activation_function = tanh
activation_function_name = "tanh"

# CIFAR-100 has 50,000 training examples, so we can experiment with some large batch sizes
batch_sizes = [64, 256, 1024]

# Train the best-performing deep network using mini-batch SGD with each batch size
for batch_size in batch_sizes:
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    logging.info(
        f"Training loader created with batch size: {batch_size}, "
        f"resulting in {len(loader_train)} mini-batches."
    )

    # Define our model
    model = BaselineDeepNetwork(activation_function=activation_function).to(device)
    model_name = f"{activation_function_name}_b={batch_size}_deep"

    # Train the model
    train_model(
        model=model,
        train_loader=loader_train,
        test_loader=loader_test,
        device=device,
        wandb_tags=[f"{activation_function_name}", f"b={batch_size}", "deep"],
        model_name=model_name,
    )

# %% ----- Optimizers: Mini-batch SGD with Momentum -----
# Pick best mini-batch size from previous step
batch_size = 64

# Define the best activation function
activation_function = tanh
activation_function_name = "tanh"

# Load data
loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Define a set of rates to use for momentum
momentum_rates = [1.5]

# Train the best-performing deep network using each rate
for momentum_rate in momentum_rates:
    # Create an instance of the model
    model = BaselineDeepNetwork(activation_function=activation_function)
    model_name = (
        f"{activation_function_name}_deep_model_b={batch_size}_alpha={momentum_rate}"
    )

    # Train the model using momentum
    train_model(
        model=model,
        train_loader=loader_train,
        test_loader=loader_test,
        device=device,
        alpha=momentum_rate,
        wandb_tags=[
            activation_function_name,
            f"b={batch_size}",
            f"alpha={momentum_rate}",
            "deep",
            "for_report",
        ],
        model_name=model_name,
    )


# %% Part 3
# ######################
# ## Skip Connections ##
# ######################

# %% ----- Extending the Model: Training -----
# Setup best options from prior steps
batch_size = 64
activation_function = tanh
activation_function_name = "tanh"
momentum_rate = 0.5

# Load data
loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Create an instance of the extended network
model = ExtendedDeepModel(activation_function=activation_function)
model_name = f"{activation_function_name}_extended_deep_model_b={batch_size}_alpha={momentum_rate}"

# Train the model using momentum
train_model(
    model=model,
    train_loader=loader_train,
    test_loader=loader_test,
    device=device,
    alpha=momentum_rate,
    patience=50,
    wandb_tags=[
        activation_function_name,
        f"b={batch_size}",
        f"alpha={momentum_rate}",
        "extended_deep",
        "for_report",
    ],
    model_name=model_name,
    track_norms=True,
)

# %% ----- Training Skip Connection #1 -----
# Setup best options from prior steps
batch_size = 64
activation_function = tanh
activation_function_name = "tanh"
momentum_rate = 0.5

# Load data
loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Create an instance of the first skip connection network
model = SkipConnectionModel1(activation_function=activation_function)
model_name = f"SkipConnection1"

# Train the model using momentum
train_model(
    model=model,
    train_loader=loader_train,
    test_loader=loader_test,
    device=device,
    alpha=momentum_rate,
    patience=50,
    wandb_tags=[
        activation_function_name,
        f"b={batch_size}",
        f"alpha={momentum_rate}",
        "skip_connection_1",
        "for_report",
    ],
    model_name=model_name,
    track_norms=True,
)

# Create an instance of the second skip connection network
model = SkipConnectionModel2(activation_function=activation_function)
model_name = f"SkipConnection2"

# Train the model using momentum
train_model(
    model=model,
    train_loader=loader_train,
    test_loader=loader_test,
    device=device,
    alpha=momentum_rate,
    patience=50,
    wandb_tags=[
        activation_function_name,
        f"b={batch_size}",
        f"alpha={momentum_rate}",
        "skip_connection_2",
        "for_report",
    ],
    model_name=model_name,
    track_norms=True,
)
