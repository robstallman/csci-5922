# %% Setup
# Imports
import matplotlib.pyplot as plt
import wandb
import torch

# Load the W&B API
api = wandb.Api()


# Define a function for visualizing training curves
def plot_loss_acc(training_curve, save_path: str = ""):
    epochs = training_curve['epochs']
    train_losses = training_curve["training_loss"]
    test_losses = training_curve["testing_loss"]
    train_accuracies = training_curve["training_accuracy"]
    test_accuracies = training_curve["testing_accuracy"]

    # Create a figure and subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    # Plot loss on the first subplot
    ax[0].plot(epochs, train_losses, label='Train set loss')
    ax[0].plot(epochs, test_losses, label="Test set loss")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)
    ax[0].legend()

    # Plot accuracy on the second subplot
    ax[1].plot(epochs, train_accuracies, label='Train set accuracy')
    ax[1].plot(epochs, test_accuracies, label="Test set accuracy")
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


label_mapping = {
    "05_momentum": "$\\alpha$" + "=0.5",
    "07_momentum": "$\\alpha$" + "=0.7",
    "09_momentum": "$\\alpha$" + "=0.9",
    "64batch_test": "b=64",
    "256batch_test": "b=256",
    "1024batch_test": "b=1024",
    "deep_baseline": "Baseline (deep)",
    "leakyReLU_test": "leaky ReLU",
    "tanh_test": "tanh",
    "extended_baseline": "Baseline (extended)",
    "skip1": "Configuration #1",
    "skip2": "Configuration #2",
    "shallow_baseline": "Baseline (2-layer)",
}

linestyle_mapping = {
    "momentum": "--",
    "batch": ":",
    "activation function": "-.",
    "baseline": "-",
}


def match_test_type(filename):
    modelname = filename.split(".csv")[0]
    if "momentum" in modelname:
        test_type = "momentum"
    elif "batch" in modelname:
        test_type = "batch"
    elif modelname == "leakyReLU_test" or modelname == "tanh_test":
        test_type = "activation function"
    else:
        test_type = "baseline"
    return test_type


def accuracy_comparison_plot(curve_dfs, label_mapping, linestyle_mapping):
    plt.figure()
    for filename, df in curve_dfs.items():
        linestyle = linestyle_mapping[match_test_type(filename)]
        label = label_mapping[filename.split(".csv")[0]]
        plt.plot(df["epochs"], df["testing_accuracy"], label=label, linestyle=linestyle)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


# %% --- Downloading --- #
# Define run alias and pathname
run_alias = "skip2"
run_path = "rost5691-cu-boulder/CSCI5922_Lab2/qu3dhtjq"

# Retrieve the run from the api
run = api.run(run_path)

# Pull the training curve
history_df = run.history()

# Add the 'epochs' column
history_df["epochs"] = history_df["_step"] + 1

# Save the df
history_df.to_csv(f"training_curves/{run_alias}.csv", index=False)


# %%
# Define Leaky ReLU activation function
def leaky_ReLU(z: torch.tensor) -> torch.tensor:
    return torch.clamp(z, min=0.1 * z)


# Define tanh activation function
def tanh(x: torch.tensor) -> torch.tensor:
    pos_exp = torch.exp(x)
    neg_exp = torch.exp(-x)
    return (pos_exp - neg_exp) / (pos_exp + neg_exp)


xvals = torch.linspace(-3, 3, 300)
tanh_yvals = tanh(xvals)
relu_yvals = leaky_ReLU(xvals)
tanh_grad = torch.gradient(tanh_yvals)[0]
relu_grad = torch.gradient(relu_yvals)[0]

plt.figure()
plt.plot(xvals, tanh_yvals, color="tab:blue", linestyle="-", label="tanh")
plt.plot(xvals, relu_yvals, color="tab:orange", linestyle="-", label="relu")
plt.plot(xvals, tanh_grad, color="tab:blue", linestyle="--", label="d(tanh)/dx")
plt.plot(xvals, relu_grad, color="tab:orange", linestyle="--", label="d(relu)/dx")
plt.legend()
plt.xlim([-1, 1])
plt.ylim([-1, 1])


# %%
import os
import pandas as pd

curve_files = [
    filename
    for filename in os.listdir("./training_curves")
    if "gradnorms" not in filename
]
curve_dfs = {
    filename: pd.read_csv(f"./training_curves/{filename}") for filename in curve_files
}

best_performances = {
    "model": [],
    "best test accuracy": [],
    "best test accuracy epoch": [],
    "best test loss": [],
    "best test loss epoch": [],
    "runtimes": [],
}
for filename, curve in curve_dfs.items():
    best_performances["model"].append(filename.split(".csv")[0])
    best_performances["best test accuracy"].append(curve["testing_accuracy"].max())
    best_accuracy_index = curve["testing_accuracy"].idxmax()
    best_performances["best test accuracy epoch"].append(best_accuracy_index + 1)
    best_performances["best test loss"].append(curve["testing_loss"].min())
    best_performances["best test loss epoch"].append(curve["testing_loss"].idxmin() + 1)
    best_performances["runtimes"].append(curve["_runtime"][best_accuracy_index])

best_performances = pd.DataFrame(best_performances)
best_performances.sort_values(
    by=["best test accuracy", "best test accuracy epoch"], ascending=False
)


# %%
filenames = [
    "shallow_baseline.csv",
    "deep_baseline.csv",
    "tanh_test.csv",
    "leakyReLU_test.csv",
]

filenames = [
    "64batch_test.csv",
    "05_momentum.csv",
    "07_momentum.csv",
    "09_momentum.csv",
]

plt.figure()
for filename in filenames:
    label = label_mapping[filename.split(".csv")[0]]
    df = curve_dfs[filename]
    plt.plot(df["epochs"], df["testing_accuracy"], label=label)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid()
# %%
# Load files
df1 = pd.read_csv("./training_curves/extended_baseline_gradnorms.csv")
df2 = pd.read_csv("./training_curves/skip1_gradnorms.csv")
df3 = pd.read_csv("./training_curves/skip2_gradnorms.csv")

# Merge dfs
df = pd.merge(
    left=df1, right=df2, on="Batch", suffixes=["_baseline", "_1"], how="outer"
)
df = pd.merge(left=df, right=df3, on="Batch", how="outer")
df.rename(
    columns={
        "Gradient L1-norm_baseline": "Extended Baseline",
        "Gradient L1-norm_1": "Configuration #1",
        "Gradient L1-norm": "Configuration #2",
    },
    inplace=True,
)
ax = df.set_index("Batch").plot()
ax.set_ylabel("Gradient L1-norm")
ax.grid()
# %%
