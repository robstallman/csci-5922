# %% Setup
# Imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Load the W&B API
api = wandb.Api()

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% --- Downloading Models --- #
# Define artifact path
artifact_alias = "trained-model:v10"
artifact_path = f"rost5691-cu-boulder/CSCI5922_Lab3/{artifact_alias}"


# Retrieve the artifact from the api
artifact = api.artifact(artifact_path)

# Download the artifact
# Should go to: ./artifacts/{artifact_alias}/baseline.pt
artifact.download()

# %%
# Load the model
class VQAModel(nn.Module):
    def __init__(
        self,
        image_droput_rate: float = 0.5,
        vocab_size: int = 1110,
        seq_len: int = 20,
        embed_dim: int = 64,
        nhead: int = 1,
        feedforward_dim: int = 2048,
        fusion_dim: int = 1024,
        n_classes: int = 595,
    ):
        super(VQAModel, self).__init__()

        # Embedding layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=seq_len, embedding_dim=embed_dim)

        # Transformer layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            batch_first=True
        )
        self.fc_question = nn.Linear(seq_len*embed_dim, fusion_dim)

        # Image layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.image_droput = nn.Dropout(p=image_droput_rate)
        self.fc_image = nn.Linear(256*5*5, fusion_dim)

        # Fusion layer
        self.fc_fusion = nn.Linear(fusion_dim, fusion_dim)

        # Final classification layer
        self.fc_output = nn.Linear(fusion_dim, n_classes)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, image, question):
        #-- Question part --#
        batch_size, seq_len = question.shape
        positions = torch.arange(0, seq_len, device=question.device).expand(batch_size, seq_len)
        x = self.embedding(question) + self.pos_embedding(positions)

        #-- Transformer block --#
        transformer_output = self.encoder_layer(x)

        #-- Image part --#
        x_image = F.relu(self.conv1(image))
        x_image = self.pool1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)
        x_image = self.image_droput(x_image)

        #-- Fusion part --#
        # Flatten image and pass through fc layer
        x_image = x_image.view(x_image.size(0), -1)
        x_image = self.fc_image(x_image)
        
        # Flatten question and pass through fc layer
        x_question = transformer_output.view(transformer_output.size(0), -1)
        x_question = self.fc_question(x_question)
        x_fusion = x_image * x_question

        # Output classification
        logits = self.fc_output(x_fusion)
        return logits

# Load the state of the saved model
if device == torch.device('cpu'):
    model_state_dict = torch.load(f"artifacts/{artifact_alias}/baseline.pt", map_location=torch.device('cpu'))
else:
    model_state_dict = torch.load(f"artifacts/{artifact_alias}/baseline.pt")

if artifact_alias == "trained-model:v4":
    model = VQAModel(
        nhead=1,
        n_classes=1
    ).to(device)
else:
    model = VQAModel(
        image_droput_rate=0.2,
        embed_dim=128,
        nhead=4,
        fusion_dim=512,
        n_classes=595,
    )
model.load_state_dict(model_state_dict, )
model.eval()
# %%
