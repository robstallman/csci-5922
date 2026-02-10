#%% Setup
# 3rd party libraries
import matplotlib.pyplot as plt
import pandas as pd

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


filename = "/Users/robstallman/Downloads/training_curves/baseline_shallow_model.csv"
df = pd.read_csv(filename)
df['train_accuracies'] = df['train_accuracies'].apply(lambda val: float(val.split("tensor(")[1].split(",")[0]))
df['test_accuracies'] = df['test_accuracies'].apply(lambda val: float(val.split("tensor(")[1].split(",")[0]))
df
# %%
plot_loss_acc(training_curve=df)

# %%
