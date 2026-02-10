# Imports
from Lab2 import *

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