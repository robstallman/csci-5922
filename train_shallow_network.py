# Imports
from Lab2 import *

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
