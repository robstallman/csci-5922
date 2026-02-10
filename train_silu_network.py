# Imports
from Lab2 import *

# Define a modified deep network, replacing the sigmoid activation function with SiLU
model = BaselineDeepNetwork(activation_function=SiLU).to(device)

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    wandb_tags=["silu_activation", "deep"],
    model_name="silu_deep",
)