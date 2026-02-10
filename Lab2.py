# %% Setup
# 3rd party imports
import copy
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
import typing
import wandb
from dotenv import load_dotenv
from torchvision import datasets
from torch.utils.data import DataLoader

# Local imports
from logs import make_logger

# Set up W&B
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
project_name = "CSCI5922_Lab2"
entity = os.getenv("WANDB_ENTITY")

# Set up logging
logger = make_logger(log_prefix="Lab2_training")

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
        x = x.view(x.size(0), -1)     # flatten
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
        self.update_parameter(self.linear1.weight, lr=lr, alpha=alpha)
        self.update_parameter(self.linear1.bias, lr=lr, alpha=alpha)
        self.update_parameter(self.linear2.weight, lr=lr, alpha=alpha)
        self.update_parameter(self.linear2.bias, lr=lr, alpha=alpha)
        self.update_parameter(self.linear3.weight, lr=lr, alpha=alpha)
        self.update_parameter(self.linear3.bias, lr=lr, alpha=alpha)

    def update_parameter(self, parameter, lr, alpha):
        # Start-up logic
        if self.first_pass:
            m = 0
            self.first_pass = False
        else:
            m = parameter.data

        # Modify the gradient with momentum
        grad = m * alpha + parameter.grad

        # Update the parameter
        grad.data -= lr * grad


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


# Define a function to evaluate a single epoch
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train: bool = True,
    lr: float = 0.001,
    alpha: float = 0.0,
) -> tuple[float, float]:
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
                model.backward(loss, lr, alpha)

            # Update running counts for loss and accuracy
            loss += loss.item()
            predictions = logits.argmax(dim=1)
            n_correct += (predictions == yb).sum()
            n_total += yb.shape[0]

    # Calculate accuracy for this epoch
    acc = n_correct / n_total

    # Return loss and accuracy for this epoch
    return loss.item(), acc.item()


# Define function for training
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: int = 0.001,
    n_epochs: int = 5000,
    print_every: int = 50,
    early_stopping: bool = True,
    patience: int = 500,
    min_delta: float = 1e-2,
    alpha: float = 0.0,
    wandb_config: dict = {},
    wandb_tags: list[str] = [],
    wand_notes: str = "",
    model_name: str = "baseline",
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
        notes=wand_notes,
        tags=wandb_tags,
        config=wandb_config,
    ) as run:
        for epoch in range(n_epochs):
            # Training over epoch
            train_loss, train_acc = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                train=True,
                lr=lr,
                alpha=alpha,
            )

            # Testing over epoch
            test_loss, test_acc = run_epoch(
                model=model,
                loader=test_loader,
                device=device,
                train=False,
                lr=lr,
                alpha=alpha,
            )

            # Save losses and accuracies for this epoch
            run.log(
                {
                    "training_loss": train_loss,
                    "testing_loss": test_loss,
                    "training_accuracy": train_acc,
                    "testing_accuracy": test_acc,
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
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# %% ----- Evaluating the Dataset Difficulty: Training -----
# Create an instance of the model
model = TwoLayerNetwork().to(device)

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    lr=0.0001,
    wandb_tags=["baseline", "shallow", "slower"],
    model_name="baseline_shallow_v2",
    wand_notes="Decrease learning rate from 1e-3 to 1e-4",
)

# %% ----- Building a Baseline Deep Network: Definitions -----

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
        self.first_pass = True  # For SGD with momentum

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

        # Layer 4
        x = x.view(x.size(0), -1)  # flatten
        x = self.linear4(x)
        x = self.activation_fn(x)

        # Layer 5
        x = self.linear5(x)
        output = softmax(x)

        return output

    def backward(self, loss: torch.tensor, lr: float, alpha: float = 0.0) -> None:
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
        self.update_parameter(self.conv1.weight, lr=lr, alpha=alpha)
        self.update_parameter(self.conv1.bias, lr=lr, alpha=alpha)
        self.update_parameter(self.conv2.weight, lr=lr, alpha=alpha)
        self.update_parameter(self.conv2.bias, lr=lr, alpha=alpha)
        self.update_parameter(self.conv3.weight, lr=lr, alpha=alpha)
        self.update_parameter(self.conv3.bias, lr=lr, alpha=alpha)
        self.update_parameter(self.linear4.weight, lr=lr, alpha=alpha)
        self.update_parameter(self.linear4.bias, lr=lr, alpha=alpha)
        self.update_parameter(self.linear5.weight, lr=lr, alpha=alpha)
        self.update_parameter(self.linear5.bias, lr=lr, alpha=alpha)

    def update_parameter(self, parameter, lr, alpha):
        # Start-up logic
        if self.first_pass:
            m = 0
            self.first_pass = False
        else:
            m = parameter.data

        # Modify the gradient with momentum
        grad = m * alpha + parameter.grad

        # Update the parameter
        grad.data -= lr * grad


# %% ----- Building a Baseline Deep Network: Training -----
# # Create an instance of the model
# model = BaselineDeepNetwork().to(device)

# # Train the model
# train_model(
#     model=model,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     device=device,
#     wandb_tags=["baseline", "deep"],
#     model_name="baseline_deep",
# )

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

# # Define a modified deep network, replacing the sigmoid activation function with tanh
# model = BaselineDeepNetwork(activation_function=tanh).to(device)

# # Train the model
# train_model(
#     model=model,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     device=device,
#     wandb_tags=["tanh_activation", "deep"],
#     model_name="tanh_deep",
# )

# %% ----- Activation Functions: Training (Pt 2) -----

# # Define a modified deep network, replacing the sigmoid activation function with SiLU
# model = BaselineDeepNetwork(activation_function=SiLU).to(device)

# # Train the model
# train_model(
#     model=model,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     device=device,
#     wandb_tags=["silu_activation", "deep"],
#     model_name="silu_deep",
# )

# %% ----- Optimizers: Mini-batch SGD -----
# # CIFAR-100 has 50,000 training examples, so we can experiment with some large batch sizes
# batch_sizes = [64, 256, 1024]

# # Train the best-performing deep network using mini-batch SGD with each batch size
# for batch_size in batch_sizes:
#     loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#     logging.info(
#         f"Training loader created with batch size: {batch_size}, "
#         f"resulting in {len(loader_train)} mini-batches."
#     )

#     # TODO: Find best activation function model
#     # Define our model
#     model = BaselineDeepNetwork(activation_function=tanh).to(device)

#     # Train the model
#     train_model(
#         model=model,
#         train_loader=loader_train,
#         test_loader=loader_test,
#         device=device,
#         wandb_tags=["tanh", f"b={batch_size}", "deep"],
#         model_name=f"tanh_b={batch_size}_deep",
#     )

# %% ----- Optimizers: Mini-batch SGD with Momentum -----
# # TODO: Pick best mini-batch size from previous step
# batch_size = 1024

# # Load data
# loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# # TODO: Define the best activation function
# activation_function = SiLU

# # Define a set of rates to use for momentum
# momentum_rates = [0.9, 0.5, 1.5]

# # Train the best-performing deep network using each rate
# for momentum_rate in momentum_rates:
#     # Create an instance of the model
#     model = BaselineDeepNetwork(activation_function=activation_function)

#     # Train the model using momentum
#     model, training_curve = train_model(
#         model=model,
#         train_loader=loader_train,
#         test_loader=loader_test,
#         device=device,
#         alpha=momentum_rate,
#     )

#     # Save the trained model
#     save_model(
#         model=model,
#         training_curve=training_curve,
#         name=f"silu_deep_model_b={batch_size}_alpha={momentum_rate}",  # TODO: CHANGE ME! I should be named after the model with the best activation function
#     )

# # %% Part 3
# ######################
# ## Skip Connections ##
# ######################

# # %% ----- Extending the Model: Definitions -----


# # Define a baseline network for deep learning
# class ExtendedDeepModel(nn.Module):

#     def __init__(
#         self,
#         input_size: int = 3072,
#         n_classes: int = 100,
#         activation_function: typing.Callable = sigmoid,
#     ):
#         super(ExtendedDeepModel, self).__init__()

#         self.input_size = input_size
#         self.n_classes = n_classes
#         self.activation_fn = activation_function
#         self.first_pass = True  # For SGD with momentum

#         # -- Layer Definitions -- #
#         # layer 1
#         self.conv1 = nn.Conv2d(
#             in_channels=3, out_channels=24, kernel_size=5, stride=1, padding=2
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

#         # layer 2
#         self.conv2 = nn.Conv2d(
#             in_channels=24, out_channels=48, kernel_size=6, stride=2, padding=0
#         )

#         # extended layers in layer 2
#         self.conv2_1 = nn.Conv2d(
#             in_channels=48, out_channels=48, kernel_size=6, stride=2, padding=9
#         )
#         self.conv2_2 = nn.Conv2d(
#             in_channels=48, out_channels=48, kernel_size=6, stride=2, padding=9
#         )
#         self.conv2_3 = nn.Conv2d(
#             in_channels=48, out_channels=48, kernel_size=6, stride=2, padding=9
#         )
#         self.conv2_4 = nn.Conv2d(
#             in_channels=48, out_channels=48, kernel_size=6, stride=2, padding=9
#         )
#         self.conv2_5 = nn.Conv2d(
#             in_channels=48, out_channels=48, kernel_size=6, stride=2, padding=9
#         )

#         # pooling for layer 2
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

#         # layer 3
#         self.conv3 = nn.Conv2d(
#             in_channels=48, out_channels=96, kernel_size=3, stride=2, padding=0
#         )
#         self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

#         # layer 4
#         # After first three layers we're left with [N_examples, 96, 2, 2]
#         self.linear4 = nn.Linear(96 * 2 * 2, 256)

#         # layer 5
#         self.linear5 = nn.Linear(256, n_classes)

#     def forward(self, x: torch.tensor) -> torch.tensor:
#         # Layer 1
#         x = self.conv1(x)
#         x = self.activation_fn(x)
#         x = self.pool1(x)

#         # Layer 2
#         x = self.conv2(x)
#         x = self.activation_fn(x)
#         x = self.conv2_1(x)
#         x = self.activation_fn(x)
#         x = self.conv2_2(x)
#         x = self.activation_fn(x)
#         x = self.conv2_3(x)
#         x = self.activation_fn(x)
#         x = self.conv2_4(x)
#         x = self.activation_fn(x)
#         x = self.conv2_5(x)
#         x = self.activation_fn(x)
#         x = self.pool2(x)

#         # Layer 3
#         x = self.conv3(x)
#         x = self.activation_fn(x)
#         x = self.pool3(x)

#         # Layer 4
#         x = x.view(x.size(0), -1)  # flatten
#         x = self.linear4(x)
#         x = self.activation_fn(x)

#         # Layer 5
#         x = self.linear5(x)
#         output = softmax(x)

#         return output

#     def backward(
#         self, loss: torch.tensor, lr: float = 0.001, alpha: float = 0.0
#     ) -> None:
#         # Reset parameter gradients
#         # NOTE: No learnable parameters for activation functions or pooling layers
#         # TODO: Account for additional layers
#         self.conv1.weight.grad = None
#         self.conv1.bias.grad = None
#         self.conv2.weight.grad = None
#         self.conv2.bias.grad = None
#         self.conv3.weight.grad = None
#         self.conv3.bias.grad = None
#         self.linear4.weight.grad = None
#         self.linear4.bias.grad = None
#         self.linear5.weight.grad = None
#         self.linear5.bias.grad = None

#         # Update gradients
#         loss.backward()

#         # Update parameters
#         self.update_parameter(self.conv1.weight, lr=lr, alpha=alpha)
#         self.update_parameter(self.conv1.bias, lr=lr, alpha=alpha)
#         self.update_parameter(self.conv2.weight, lr=lr, alpha=alpha)
#         self.update_parameter(self.conv2.bias, lr=lr, alpha=alpha)
#         self.update_parameter(self.conv3.weight, lr=lr, alpha=alpha)
#         self.update_parameter(self.conv3.bias, lr=lr, alpha=alpha)
#         self.update_parameter(self.linear4.weight, lr=lr, alpha=alpha)
#         self.update_parameter(self.linear4.bias, lr=lr, alpha=alpha)
#         self.update_parameter(self.linear5.weight, lr=lr, alpha=alpha)
#         self.update_parameter(self.linear5.bias, lr=lr, alpha=alpha)

#     def update_parameter(self, parameter, lr, alpha):
#         # Start-up logic
#         if self.first_pass:
#             m = 0
#             self.first_pass = False
#         else:
#             m = parameter.data

#         # Modify the gradient with momentum
#         grad = m * alpha + parameter.grad

#         # Update the parameter
#         grad.data -= lr * grad
