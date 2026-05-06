#----- Setup -----#
# Imports
import copy
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import spacy
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as v2
import typing
import wandb
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional

# Set up W&B
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
project_name = "CSCI5922_Lab3"
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
    ROOT_DATA_PATH='/content/drive/My Drive/Colab Notebooks/CSCI 5922/data/Lab 3/'
else:
    ROOT_DATA_PATH='/Users/robstallman/Library/CloudStorage/GoogleDrive-robstallman3@gmail.com/My Drive/Colab Notebooks/CSCI 5922/data/Lab 3/'

# Run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----- Definitions -----#
# Define paths to raw data
TRAIN_PATH = os.path.join(ROOT_DATA_PATH, "train")
VAL_PATH = os.path.join(ROOT_DATA_PATH, "val")
TEST_PATH = os.path.join(ROOT_DATA_PATH, "test")
ANNOTATION_PATH_TRAIN = os.path.join(ROOT_DATA_PATH, "Annotations/train.json")
ANNOTATION_PATH_VAL = os.path.join(ROOT_DATA_PATH, "Annotations/val.json")
ANNOTATION_PATH_TEST = os.path.join(ROOT_DATA_PATH, "Annotations/test.json")

# Define paths to processed data
TOK2IDX_PATH = os.path.join(ROOT_DATA_PATH, "tok2idx.json")
IDX2TOK_PATH = os.path.join(ROOT_DATA_PATH, "idx2tok.json")
VOCAB_COUNTER_PATH = os.path.join(ROOT_DATA_PATH, "vocab_counter.json")
EMBEDDED_QUESTIONS_PATH_TRAIN = os.path.join(ROOT_DATA_PATH, "questions_train.pt")
EMBEDDED_QUESTIONS_PATH_VAL = os.path.join(ROOT_DATA_PATH, "questions_val.pt")
EMBEDDED_QUESTIONS_PATH_TEST = os.path.join(ROOT_DATA_PATH, "questions_test.pt")
NAME2ID_PATH = os.path.join(ROOT_DATA_PATH, "name2id.json")
ID2NAME_PATH = os.path.join(ROOT_DATA_PATH, "id2name.json")
ENCODED_ANSWERS_PATH_TRAIN = os.path.join(ROOT_DATA_PATH, "answers_train.pt")
ENCODED_ANSWERS_PATH_VAL = os.path.join(ROOT_DATA_PATH, "answers_val.pt")


# Define other constants
TRAIN_PCT = 0.487   # Percentage of available training data to use
VIZWIZ_MEAN = [0.49117913842201233, 0.43745487928390503, 0.3850884735584259]
VIZWIZ_STD = [0.2856180667877197, 0.2747800052165985, 0.2757737636566162]
THRESHOLD = 0.50             # minimum answer frequency within a sample
FREQUENCY_CUTOFF = 2         # Minimum number of times that an answer must appear to be included in corpus
nlp = spacy.load('en_core_web_sm')

class VizWizLoader(torch.utils.data.Dataset):
    def __init__(self, strFolder: str, strAnnotationPath: str, fDataPercentage: float = 1.0,
                 tTransform: Callable[[torch.tensor], torch.tensor] = None, 
                 embedded_questions: Optional[torch.tensor] = None,
                 embedded_answers: Optional[torch.tensor] = None) -> None:
        '''
        strFolder: path to unzip'd folder of VizWiz images
        strLabelPath: path to .json file containing the annotations
        fDataPercentage: percentage of available samples to use. Must be normalized between 0.0 and 1.0. Default: 1.0
        tTransform: optional place to connect PyTorch image transformations. Default: converts images to 3x224x224 tensors
        For the train and val splits, returns tuples of the form:
            (image, question text, binary label, answer texts)
        Otherwise, returns tuples of the form:
            (image, question text)
        '''
        # Define paths
        self.strFolder = strFolder
        if self.strFolder[-1] != "/": self.strFolder += "/"
        vecPaths = os.listdir(self.strFolder)
        self.strPrefix = vecPaths[0].split("_")[1]

        # Load annotations
        with open(strAnnotationPath, "r") as f:
            self.vecAnnos = json.load(f)
        
        # Modify length based on fDataPercentage
        self.iN = int(fDataPercentage * len(self.vecAnnos))

        # Define transformations for images
        base_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True),
            v2.RandomCrop(224)
            ])
        base_transform_undersized = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True), v2.Resize((224, 224))])
        if tTransform is None:
            self.tTransform = base_transform
            self.tTransformUndersized = base_transform_undersized
        else:
            self.tTransform = v2.Compose(base_transform.transforms + tTransform.transforms)
            self.tTransformUndersized = v2.Compose(base_transform_undersized.transforms + tTransform.transforms)
        
        # Assign Define questions and answers
        self.embedded_questions = embedded_questions
        self.embedded_answers = embedded_answers

        return

    def __len__(self) -> int: return self.iN

    def __getitem__(self, idx: int) -> tuple:
        if idx >= self.iN:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.iN}")
        
        strPath = self.strFolder + self.vecAnnos[idx]["image"]
        imX = Image.open(strPath)
        w, h = imX.size
        if w >= 224 and h >= 224: tX = self.tTransform(imX)
        else: tX = self.tTransformUndersized(imX)
        annotations = self.vecAnnos[idx]

        if self.embedded_questions is None:
            # Return raw annotations
            if self.strPrefix == "test":
                return tX, self.vecAnnos[idx]["question"]
            else:
                return tX, self.vecAnnos[idx]["question"], self.vecAnnos[idx]["answerable"], self.vecAnnos[idx]["answers"]
        else:
            # Return embedded annotations
            question = self.embedded_questions[idx]
            if self.embedded_answers is None:
                return tX, question
            else:
                answer = self.embedded_answers[idx]
                human_answers = [entry['answer'] for entry in annotations['answers']]
                return tX, question, annotations["answerable"], answer, human_answers

def load_datasets(mean = VIZWIZ_MEAN, std = VIZWIZ_STD, fDataPercentage=TRAIN_PCT):
    # Compute mean and standard deviation if they are not passed
    if mean is None and std is None:
        print(f"No mean, std available. Calculating from training data...")
        train_set_raw = VizWizLoader(
            strFolder=TRAIN_PATH,
            strAnnotationPath=ANNOTATION_PATH_TRAIN,
            fDataPercentage=fDataPercentage
        )
        print(f"Loaded training dataset")

        # Use data loader on raw data
        train_loader_raw = DataLoader(train_set_raw, batch_size=64, shuffle=False)
        print(f"Loaded training DataLoader")

        # Compute per-channel mean/std for the train set
        sum_ = torch.zeros(3)
        sum_sq = torch.zeros(3)
        n_pixels = 0

        # Compute batch-by-batch to prevent runtime crash
        print("Calculating mean, std...")
        for x, _, _, _ in train_loader_raw:
            sum_ += x.sum(dim=(0,2,3))
            sum_sq += (x ** 2).sum(dim=(0,2,3))
            n_pixels += x.size(0) * x.size(2) * x.size(3)

        mean = sum_ / n_pixels
        std = (sum_sq / n_pixels - mean ** 2).sqrt()
    else:
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)

    # Print results
    
    print(f"Vizwiz mean: {mean.tolist()}")
    print(f"Vizwiz std: {std.tolist()}")

    # Define transformations
    tfms = v2.Compose([
        v2.Normalize(mean, std)
    ])

    # Load datasets with normalization
    train_set = VizWizLoader(
        strFolder=TRAIN_PATH,
        strAnnotationPath=ANNOTATION_PATH_TRAIN,
        fDataPercentage=fDataPercentage,
        tTransform=tfms,
    )
    val_set = VizWizLoader(
        strFolder=VAL_PATH,
        strAnnotationPath=ANNOTATION_PATH_VAL,
        tTransform=tfms
    )
    test_set = VizWizLoader(
        strFolder=TEST_PATH,
        strAnnotationPath=ANNOTATION_PATH_TEST,
        tTransform=tfms
    )

    return train_set, val_set, test_set

def batch_tokenize(samples, nlp, batch_size=512):
    texts = [sample[1] for sample in samples]
    tokenized = []

    docs = nlp.pipe(texts, batch_size=batch_size, disable=['ner', 'parser'])
    print(f"Loaded docs from {len(texts)} text samples")

    for doc in docs:
        tokens = [tok.lemma_.lower() for tok in doc if not tok.is_punct and not tok.is_space and not tok.is_stop]
        tokenized.append(tokens)

    return tokenized

def build_vocab(token_list, min_frequency=2, max_vocab_size: typing.Optional[int] = None):
    counter = Counter()
    for sample in token_list:
        counter.update(sample)
    
    special_tokens = ["<START>", "<END>", "<PAD>", "<UNK>"]
    tok2idx = {tok:idx for idx, tok in enumerate(special_tokens)}
    for tok, counts in counter.most_common():
        if counts >= min_frequency:
            tok2idx[tok] = len(tok2idx)
    
    if max_vocab_size is not None:
        tok2idx = {k:v for idx, (k,v) in enumerate(tok2idx.items()) if idx < max_vocab_size}
    idx2tok = {idx:tok for tok, idx in tok2idx.items()}
    
    return tok2idx, idx2tok, counter

def embed_dataset(tokenized_samples, tok2idx, max_len=20):
    UNK, PAD = tok2idx["<UNK>"], tok2idx["<PAD>"]
    embedded = []

    for sample in tokenized_samples:
        # Get ids
        ids = [tok2idx.get(tok, UNK) for tok in sample]
        # Pad ids
        ids = ids[:max_len] + [PAD] * max(0, max_len - len(ids))
        embedded.append(ids)
    
    return embedded

def get_sample_corpus_entries(sample: list) -> list:
    """
    For a single sample, if the prompt is 'answerable', return a list of answers that:
        1) Are indicated with 'answer_confidence' = 'yes'
        2) Have at least 3 reviewers agreeing

    If the prompt is unanswerable, or if no answers are left after filtering, return an empty list

    Parameters
    ----------
    sample : list
        A sample in the VizWiz training set

    Returns
    -------
    list
        List of answer strings that meet the requirements
    """
    answerable, answer_list = sample[2], sample[3]
    if answerable:
        high_confidence_answers = [entry['answer'] for entry in answer_list if entry['answer_confidence'] == 'yes']
        total = len(high_confidence_answers)
        counts = Counter(high_confidence_answers)
        if total == 0:
            return []
        else:
            return [ans for ans, answer_counts in counts.most_common() if min(answer_counts/3, 1) == 1]
    else:
        return []

def encode_sample(sample: list, answer_index: dict[str, int]) -> torch.Tensor:
    num_classes = len(answer_index)
    target = torch.zeros(num_classes, dtype=torch.float)

    sample_answers = get_sample_corpus_entries(sample)

    # Encode all real answers that cleared the threshold and are in the corpus
    if len(sample_answers) == 0:
        target[answer_index['unanswerable']] = 1.0
    else:
        for answer in sample_answers:
            if answer in answer_index.keys():
                target[answer_index[answer]] = 1.0
            else:
                target[answer_index['other_categories']] = 1.0

    return target

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
        x_fusion = F.relu(self.fc_fusion(x_fusion))

        # Output classification
        logits = self.fc_output(x_fusion)
        return logits

def save_model(
    model: nn.Module,
    name: str,
    root_path: str = ROOT_DATA_PATH
):
    # Create a directory for models if it doesn't yet exist
    if not os.path.exists(os.path.join(root_path, "models")):
        os.mkdir(os.path.join(root_path, "models"))

    # Save the model to the directory
    filepath = os.path.join(root_path, "models", f"{name}.pt")
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to: {filepath}")

    # Return the filepath
    return filepath

def calculate_batch_accuracy(predicted_answers, ground_truth_labels):
    accuracies = []
    batch_size = len(predicted_answers)

    for idx in range(batch_size):
        # Normalize answers to lowercase
        predicted_answer = predicted_answers[idx].lower()
        human_answers = [answer[idx].lower() for answer in ground_truth_labels]

        # Get the counts of humans who agreed with predicted answer
        answer_counts = Counter(human_answers)
        humans_agreed = answer_counts.get(predicted_answer, 0)

        # Calculate accuracy as min(# humans who agreed/3, 1)
        sample_accuracy = min(humans_agreed/3, 1)
        accuracies.append(sample_accuracy)

    accuracy = np.mean(accuracies)
    return accuracy

def eval_accuracy_answer(
    model: nn.Module,
    dataset: VizWizLoader,
    category_id2name: dict,
    eval_every: int = 1,
    eval_batch_size: int = 1
):
    # Set the model to evaluation mode
    model.eval()

    # Initial values for accuracy
    tracked_accuracies = []

    # Get the device from the model
    device = model.device

    # Create a dataloader
    loader = DataLoader(dataset=dataset, batch_size=eval_batch_size, shuffle=False)

    with torch.no_grad():
        for idx, (image_b, question_b, _, _, human_answer_b) in enumerate(loader):
            if (idx % eval_every) == (eval_every - 1):
                # Transfer to device
                image_b = image_b.to(device)
                question_b = question_b.to(device)

                # Get model outputs
                logits = model(image_b, question_b)

                # Recover prediction
                probs = torch.sigmoid(logits)
                preds = probs.argmax(dim=1).tolist()    # Use most-probable answer
                answers = [category_id2name[pred] for pred in preds]

                # Evaluate accuracy
                batch_accuracy = calculate_batch_accuracy(answers, human_answer_b)
                tracked_accuracies.append(batch_accuracy)
    
    accuracy = np.mean(tracked_accuracies)
    return accuracy
               
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    train: bool = True,
):
    # Set the model mode – either training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # Initial values for tracking loss over the epoch
    running_loss = 0.0

    # Get the device from the model
    device = model.device

    # Set context based on model mode
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        # Iterate through the batches in the loader
        for image_b, question_b, _, answer_b, _ in loader:
            # Transfer to device
            image_b = image_b.to(device)
            question_b = question_b.to(device)
            answer_b = answer_b.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # -- Forward pass -- #
            # Get model output
            logits = model(image_b, question_b)

            # Evaluate loss
            loss = loss_fn(logits, answer_b)

            # -- Backward pass -- #
            if train:
                # Update gradients
                loss.backward()

                # Adjust learning weights
                optimizer.step()

            # Update running counts for loss and accuracy
            running_loss += loss.item()

    # Calculate average batch loss
    running_loss = running_loss / len(loader)

    # Return loss and accuracy for this epoch
    return running_loss

def eval_accuracy_binary(
    model: nn.Module,
    dataset: VizWizLoader,
    eval_every: int = 1,
    eval_batch_size: int = 1
):
    # Set the model to evaluation mode
    model.eval()

    # Initial values for accuracy
    tracked_accuracies = []

    # Get the device from the model
    device = model.device

    # Create a dataloader with one batch
    loader = DataLoader(dataset=dataset, batch_size=eval_batch_size, shuffle=False)

    with torch.no_grad():
        for idx, (image_b, question_b, target, _, _) in enumerate(loader):
            if (idx % eval_every) == (eval_every - 1):
                # Transfer to device
                image_b, question_b, target = image_b.to(device), question_b.to(device), target.to(device)

                # Get model output
                logits = model(image_b, question_b)

                # Inference: recover prediction
                probs = torch.sigmoid(logits).squeeze()
                preds = (probs >= 0.5).long()
                target = target.long()

                # Calculate categories
                TP = ((preds == 1) & (target == 1)).sum().item()
                FP = ((preds == 1) & (target == 0)).sum().item()
                TN = ((preds == 0) & (target == 0)).sum().item()
                FN = ((preds == 0) & (target == 1)).sum().item()

                # Calculate accuracy
                accuracy = (TP + TN) / (TP + FP + TN + FN)
                
                tracked_accuracies.append(accuracy)
    
    final_accuracy = np.mean(tracked_accuracies)
    return final_accuracy

def run_epoch_binary(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    train: bool = True,
):
    # Set the model mode – either training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # Initial values for tracking loss over the epoch
    running_loss = 0.0

    # Get the device from the model
    device = model.device

    # Set context based on model mode
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        # Iterate through the batches in the loader
        for image_b, question_b, target_b, _, _ in loader:
            # Reshape binary labels and cast as floats
            target_b = target_b.reshape(target_b.size(0), -1).to(torch.float)
            
            # Transfer to device
            image_b = image_b.to(device)
            question_b = question_b.to(device)
            target_b = target_b.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # -- Forward pass -- #
            # Get model output
            logits = model(image_b, question_b)

            # Evaluate loss
            loss = loss_fn(logits, target_b)

            # -- Backward pass -- #
            if train:
                # Update gradients
                loss.backward()

                # Adjust learning weights
                optimizer.step()

            # Update running counts for loss and accuracy
            running_loss += loss.item()

    # Calculate average batch loss
    running_loss = running_loss / len(loader)

    # Return loss and accuracy for this epoch
    return running_loss

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_set: VizWizLoader,
    eval_batch_size: int,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    n_epochs: int = 100,
    category_id2name: dict = {},
    early_stopping: bool = True,
    patience: int = 25,
    min_delta: float = 1e-2,
    eval_every: int = 1,
    epoch_eval_every: int = 1,
    print_every: int = 1,
    wandb_config: dict = {},
    wandb_tags: list[str] = [],
    wandb_notes: str = "",
    model_name: str = "baseline",
    task: str = "answer-prediction"
) -> None:
    # Early stopping setup
    best_val_accuracy = -1*float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    no_improvement_count = 0

    # W&B setup
    wandb_config["task"] = task
    wandb_config["epochs"] = n_epochs
    wandb_config["batch_size"] = train_loader.batch_size
    wandb_config["early_stopping"] = early_stopping
    wandb_config["patience"] = patience
    wandb_config["min_delta"] = min_delta

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
            if task == "answer-prediction":
                train_loss = run_epoch(
                    model=model,
                    loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    train=True,
                )
            elif task == "binary-prediction":
                train_loss = run_epoch_binary(
                    model=model,
                    loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    train=True
                )
            else:
                raise RuntimeError(f"Unknown task: {task}")

            # Evaluation over a subset of the epoch
            if (epoch % epoch_eval_every) == (epoch_eval_every - 1):
                if task == "answer-prediction":
                    val_loss = run_epoch(
                        model=model,
                        loader=val_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        train=False,
                    )
                    val_accuracy = eval_accuracy_answer(
                        model=model,
                        dataset=val_set,
                        category_id2name=category_id2name,
                        eval_every=eval_every,
                        eval_batch_size=eval_batch_size
                    )
                elif task == "binary-prediction":
                    val_loss = run_epoch_binary(
                        model=model,
                        loader=val_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        train=False,
                    )
                    val_accuracy = eval_accuracy_binary(
                        model=model,
                        dataset=val_set,
                        eval_every=eval_every,
                        eval_batch_size=eval_batch_size
                    )
                else:
                    raise RuntimeError(f"Unknown task: {task}")
            else:
                val_loss, val_accuracy = np.nan, np.nan

            # Save losses for this epoch
            run.log(
                {
                    "training_loss": train_loss,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy
                }
            )

            # Print the losses periodically
            if epoch % print_every == (print_every - 1):
                print(
                    f"Epoch {epoch+1}/{n_epochs}\n"
                    f"--------------------------\n"
                    f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}\n"
                    f"Val accuracy: {val_accuracy:.4f}\n"
                )

            # Early stopping logic
            if early_stopping:
                # Improvement means val_accuracy improved by at least min_delta
                if val_accuracy > best_val_accuracy + min_delta:
                    best_val_accuracy = val_accuracy
                    best_model_weights = copy.deepcopy(model.state_dict())
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        break

        # Reload best model
        if early_stopping:
            model.load_state_dict(best_model_weights)

        # Upload the best model as an artifact
        model_filepath = save_model(model=model, name=model_name)
        run.log_artifact(model_filepath, name="trained-model", type="model")

    return

def save_predictions(preds, root_path: str, filename: str) -> None:
    prediction_directory = os.path.join(root_path, "predictions")
    
    # Ensure directory exists
    if not os.path.exists(prediction_directory):
        os.mkdir(prediction_directory)
    
    # Save the predictions
    prediction_filepath = os.path.join(prediction_directory, filename)
    if filename.endswith(".pkl"):
        torch.save(preds, prediction_filepath)
    else:
        with open(prediction_filepath, 'w') as f:
            json.dump(preds, f, indent=4)
    print(f"Saved predictions to {prediction_filepath}")

def test_model_binary(
    model: nn.Module,
    samples: list
):
    # Get the device from the model
    device = model.device

    # Unpack the images and questions from the samples list
    image_b, question_b = samples[0], samples[1]

    # Transfer to device
    image_b, question_b = image_b.to(device), question_b.to(device)

    # Get model output
    logits = model(image_b, question_b)

    # Inference: recover prediction
    probs = torch.sigmoid(logits).squeeze()
    preds = (probs >= 0.5).long()

    # Reshape predictions
    preds = preds.view(preds.size(0))

    return preds
    
def test_model_answer(
    model: nn.Module,
    samples: list,
    category_id2name: dict
) -> list[str]:
    # Get the device from the model
    device = model.device

    # Unpack the images and questions from the samples list
    image_b, question_b = samples[0], samples[1]

    # Transfer to device
    image_b, question_b = image_b.to(device), question_b.to(device)

    # Get model output
    logits = model(image_b, question_b)
    
    # Recover prediction
    probs = torch.sigmoid(logits)
    preds = probs.argmax(dim=1).tolist()    # Use most-probable answer
    answers = [category_id2name[pred] for pred in preds]

    return answers

#----- Dataset Processing -----#
### Load train, val, and test datasets 
# Note: Image processing is built in to this step 
train_set, val_set, test_set = load_datasets()
print("Unembedded datasets are loaded with lengths:")
print(f"Training: {len(train_set)}")
print(f"Validation: {len(val_set)}")
print(f"Testing: {len(test_set)}")

### Process questions

# Look for previously-embedded questions
try:
    questions_train = torch.load(EMBEDDED_QUESTIONS_PATH_TRAIN)
    questions_train = questions_train.to(dtype=torch.long)
    questions_val = torch.load(EMBEDDED_QUESTIONS_PATH_VAL)
    questions_val = questions_val.to(dtype=torch.long)
    questions_test = torch.load(EMBEDDED_QUESTIONS_PATH_TEST)
    questions_test = questions_test.to(dtype=torch.long)

    with open(TOK2IDX_PATH, 'r') as f:
        tok2idx = json.load(f)
    with open(IDX2TOK_PATH, 'r') as f:
        idx2tok = json.load(f)
        idx2tok = {int(k) if k.isdigit() else k:v for k,v in idx2tok.items()}
    with open(VOCAB_COUNTER_PATH, 'r') as f:
        counter_dict = json.load(f)
        counter = Counter(counter_dict)  
except FileNotFoundError:
    print("Processing questions from scratch...")
    # Tokenize the train, val, and test datasets
    train_tokens = batch_tokenize(train_set, nlp)
    print(f"Tokenized {len(train_tokens)} training samples")
    val_tokens = batch_tokenize(val_set, nlp)
    print(f"Tokenized {len(val_tokens)} validation samples")
    test_tokens = batch_tokenize(test_set, nlp)
    print(f"Tokenized {len(test_tokens)} test samples")

    # Build vocabulary from the training set
    tok2idx, idx2tok, counter = build_vocab(train_tokens)
    print(f"Vocab created with {len(tok2idx)} entries")
    
    # Save vocab
    with open(TOK2IDX_PATH, 'w') as f:
        json.dump(tok2idx, f, indent=4)
    with open(IDX2TOK_PATH, 'w') as f:
        json.dump(idx2tok, f, indent=4)
    with open(VOCAB_COUNTER_PATH, 'w') as f:
        json.dump(counter, f, indent=4)

    # Encode the train, val, and test questions
    questions_train = torch.tensor(embed_dataset(train_tokens, tok2idx), dtype=torch.long)
    questions_val = torch.tensor(embed_dataset(val_tokens, tok2idx), dtype=torch.long)
    questions_test = torch.tensor(embed_dataset(test_tokens, tok2idx), dtype=torch.long)

    # Save embedded questions
    torch.save(questions_train, EMBEDDED_QUESTIONS_PATH_TRAIN)
    torch.save(questions_val, EMBEDDED_QUESTIONS_PATH_VAL)
    torch.save(questions_test, EMBEDDED_QUESTIONS_PATH_TEST)

print("Training questions embedded with shape", questions_train.shape)
print("Validation questions embedded with shape", questions_val.shape)
print("Test questions embedded with shape", questions_test.shape)

### Process answers

# Look for previously-embedded answers
try:
    answers_train = torch.load(ENCODED_ANSWERS_PATH_TRAIN)
    answers_val = torch.load(ENCODED_ANSWERS_PATH_VAL)

    with open(NAME2ID_PATH, 'r') as f:
        category_name2id = json.load(f)
    with open(ID2NAME_PATH, 'r') as f:
        category_id2name = json.load(f)
        category_id2name = {int(k) if k.isdigit() else k:v for k,v in category_id2name.items()}
except FileNotFoundError:
    print("Processing answers from scratch...")
    # Generate a list of qualified answers
    all_qualified_answers = []
    for idx, training_sample in enumerate(train_set):
        if (idx % 50) == 49:
            print(f"Working on sample: {idx+1}/{train_set.iN}")
        sample_answers = get_sample_corpus_entries(training_sample)
        if sample_answers:
            all_qualified_answers += sample_answers

    # Take all answers meeting the minimum frequency cutoff
    qualified_answer_counts = Counter(all_qualified_answers)
    answer_corpus = [ans for ans, counts in qualified_answer_counts.most_common() if counts >= FREQUENCY_CUTOFF]
    n_answers = len(answer_corpus)
    print(f"Answer corpus contains {n_answers} real answers")

    # Create categories for answer corpus
    category_name2id = {ans:idx for idx, ans in enumerate(answer_corpus)}
    category_id2name ={idx:ans for idx, ans in enumerate(answer_corpus)}
    category_id2name[n_answers] = 'other_categories'        # Bucket for answers not in the corpus
    category_name2id['other_categories'] = n_answers

    # Save categories
    with open(NAME2ID_PATH, 'w') as f:
        json.dump(category_name2id, f)
    with open(ID2NAME_PATH, 'w') as f:
        json.dump(category_id2name, f)

    # Encode the train and val answers (no test answers in dataset)
    answers_train = torch.stack([encode_sample(sample, category_name2id) for sample in train_set], dim=0)
    answers_val = torch.stack([encode_sample(sample, category_name2id) for sample in val_set], dim=0)
    
    # Save encoded answers
    torch.save(answers_train, ENCODED_ANSWERS_PATH_TRAIN)
    torch.save(answers_val, ENCODED_ANSWERS_PATH_VAL)

print("Training answers encoded as categories with shape", answers_train.shape)
print("Validation answers encoded as categories with shape", answers_val.shape)

### Build datasets with embedded questions and answers
tfms = v2.Compose([
        v2.Normalize(VIZWIZ_MEAN, VIZWIZ_STD)
    ])
training_set = VizWizLoader(
    strFolder=TRAIN_PATH,
    strAnnotationPath=ANNOTATION_PATH_TRAIN,
    fDataPercentage=0.487,
    tTransform=tfms,
    embedded_questions=questions_train,
    embedded_answers=answers_train
    )
validation_set = VizWizLoader(
    strFolder=VAL_PATH,
    strAnnotationPath=ANNOTATION_PATH_VAL,
    tTransform=tfms,
    embedded_questions=questions_val,
    embedded_answers=answers_val
    )
testing_set = VizWizLoader(
    strFolder=TEST_PATH,
    strAnnotationPath=ANNOTATION_PATH_TEST,
    tTransform=tfms,
    embedded_questions=questions_test,
    )
print("Finished preprocessing steps")

#------------- Model Training: Binary Classification Task -------------#
# Define hyperparameters
task='binary-prediction'
n_classes = 1
alpha = 0.9
batch_sizes = [512, 1024]
learning_rates = [0.01, 0.001]
n_heads = [1, 4]

for batch_size in batch_sizes:
    # Set up DataLoaders
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # Run experiments
    for nhead in n_heads:
        for lr in learning_rates:
            # Define model, loss function, and optimizer
            model = VQAModel(
                image_droput_rate=0.5,
                embed_dim=64,
                nhead=nhead,
                feedforward_dim=2048,
                fusion_dim=1024,
                n_classes=n_classes
            ).to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=alpha)

            # Train model
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_set=validation_set,
                eval_batch_size=len(validation_set),
                loss_fn=loss_fn,
                optimizer=optimizer,
                n_epochs=9,
                category_id2name=category_id2name,
                early_stopping=False,
                eval_every=1,
                epoch_eval_every=3,
                wandb_config={
                    "lr":lr,
                    "alpha":alpha,
                    "nheads": nhead,
                },
                wandb_tags=['prototyping', task],
                task=task
            )

#------------- Model Training: Answer Prediction Task -------------#
# Define hyperparameters
task='answer-prediction'
n_classes = 595
batch_size = 512
lr = 0.01
alpha = 0.9
nhead = 4
fusion_dims = [512, 256]
embed_dims = [32, 128]
dropout_rates = [0.2, 0.5]

# Set up DataLoaders
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# Run experiments
for fusion_dim in fusion_dims:
    for embed_dim in embed_dims:
        for dropout_rate in dropout_rates:
            # Define model, loss function, and optimizer
            model = VQAModel(
                image_droput_rate=dropout_rate,
                embed_dim = embed_dim,
                nhead=nhead,
                fusion_dim=fusion_dim,
                n_classes=n_classes
            ).to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=alpha)

            # Train model
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_set=validation_set,
                eval_batch_size=len(validation_set),
                loss_fn=loss_fn,
                optimizer=optimizer,
                n_epochs=9,
                category_id2name=category_id2name,
                early_stopping=False,
                eval_every=1,
                epoch_eval_every=3,
                wandb_config={
                    "lr":lr,
                    "alpha":alpha,
                    "nheads":nhead,
                    "fusion_dim":fusion_dim,
                    "embed_dim":embed_dim,
                    "dropout_rate":dropout_rate
                },
                wandb_tags=['prototyping', task],
                task=task
            )

# Load pre-trained features
CLIP_image_train = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_train_CLIP_Image.pkl"))
CLIP_image_val = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_val_CLIP_Image.pkl"))
CLIP_image_test = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_test_CLIP_Image.pkl"))
CLIP_text_train = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_train_CLIP_Text.pkl"))
CLIP_text_val = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_val_CLIP_Text.pkl"))
CLIP_text_test = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_test_CLIP_Text.pkl"))

# Limit training size
CLIP_image_train = CLIP_image_train[:10000]
CLIP_text_train = CLIP_text_train[:10000]

class CLIP_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        strAnnotationPath: str,
        embedded_images: torch.tensor,
        embedded_questions: torch.tensor,
        embedded_answers: Optional[torch.tensor] = None) -> None:
        '''
        strFolder: path to unzip'd folder of VizWiz images
        strAnnotationPath: path to .json file containing the annotations
        embedded_images: tensor of pre-trained images (from CLIP)
        embedded_questions: tensor of pre-trained questions (from CLIP)
        embedded_answers: tensors of externally embedded answers (not from CLIP)

        For the train and val splits, returns tuples of the form:
            (image, question text, binary label, answer texts)
        Otherwise, returns tuples of the form:
            (image, question text)
        '''
        # Load annotations
        with open(strAnnotationPath, "r") as f:
            self.vecAnnos = json.load(f)
        
        # Assign images, questions and answers
        self.embedded_images = embedded_images
        self.embedded_questions = embedded_questions
        self.embedded_answers = embedded_answers

        # Define length
        self.iN = self.embedded_images.size(0)

        return

    def __len__(self) -> int: return self.iN

    def __getitem__(self, idx: int) -> tuple:
        # Handle invalid requests
        if idx >= self.iN:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.iN}")
        
        # Load embedded question and image for this sample
        question = self.embedded_questions[idx]
        image = self.embedded_images[idx]
        
        # Return embedded annotations
        if self.embedded_answers is None:
            return image, question
        else:
            # Get annotations for this sample
            annotations = self.vecAnnos[idx]
            answer = self.embedded_answers[idx]
            human_answers = [entry['answer'] for entry in annotations['answers']]

            return image, question, annotations["answerable"], answer, human_answers

# Load pre-trained datasets
CLIP_training_set = CLIP_dataset(
    strAnnotationPath=ANNOTATION_PATH_TRAIN,
    embedded_images=CLIP_image_train,
    embedded_questions=CLIP_text_train,
    embedded_answers=answers_train
)
CLIP_validation_set = CLIP_dataset(
    strAnnotationPath=ANNOTATION_PATH_VAL,
    embedded_images=CLIP_image_val,
    embedded_questions=CLIP_text_val,
    embedded_answers=answers_val
)
CLIP_testing_set = CLIP_dataset(
    strAnnotationPath=ANNOTATION_PATH_TEST,
    embedded_images=CLIP_image_test,
    embedded_questions=CLIP_text_test,
)
print("Finished loading pre-processed features")

# Define model
class LightweightModel(nn.Module):
    def __init__(
        self,
        nhead,
        feedforward_dim,
        dropout_rate,
        n_classes
        ):
        super(LightweightModel, self).__init__()

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout_rate,
            batch_first=True)
        
        # Fully-connected sequence
        self.fc_sequence = nn.Sequential(
            nn.Linear(512*512, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_classes)
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, question, image):
        # Instead of embedding, try the method of CLIP
        x_fusion = torch.bmm(question.unsqueeze(2), image.unsqueeze(1))

        # Transformer layer
        out = self.encoder_layer(x_fusion)

        # Reshape output
        out = out.view(out.size(0), -1)

        # Fully-connected sequence
        logits = self.fc_sequence(out)

        return logits

class SimpleLightweightModel(nn.Module):
    def __init__(
        self,
        n_classes
        ):
        super(SimpleLightweightModel, self).__init__()
        
        # Fully-connected sequence
        if n_classes == 1:
            self.fc_sequence = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128,16),
                nn.ReLU(),
                nn.Linear(16, n_classes)
            )
        else:
            self.fc_sequence = nn.Sequential(
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Linear(1024, n_classes)
            )

    
    def forward(self, question, image):
        # Simply combine
        x_fusion = question + image

        # Fully-connected sequence
        logits = self.fc_sequence(x_fusion)

        return logits

#------------- Model Training: Binary Classification Task -------------#
# Select earnest-bird-17 settings:
# Highest accuracy (tied), lowest losses (train and val)

# Define hyperparameters
task='binary-prediction'
n_classes = 1
alpha = 0.9
batch_size = 1024
lr = 0.01
nhead = 1
feedforward_dims = [512, 1024]

# Set up DataLoaders
train_loader = DataLoader(CLIP_training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CLIP_validation_set, batch_size=batch_size, shuffle=False)

# Run experiments
for feedforward_dim in feedforward_dims:
    # Define model, loss function, and optimizer
    model = LightweightModel(
        nhead=nhead,
        feedforward_dim=feedforward_dim,
        dropout_rate=0.1,
        n_classes=n_classes
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=alpha)

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_set=CLIP_validation_set,
        eval_batch_size=len(CLIP_validation_set),
        loss_fn=loss_fn,
        optimizer=optimizer,
        n_epochs=9,
        category_id2name=category_id2name,
        early_stopping=False,
        eval_every=1,
        epoch_eval_every=3,
        wandb_config={
            "lr":lr,
            "alpha":alpha,
            "nheads": nhead,
            "feedforward_dim":feedforward_dim
        },
        wandb_tags=['prototyping', task, "CLIP"],
        task=task
    )

#------------- Model Training: Answer Prediction Task -------------#
# Use parameters from breezy-dew-23:
# Highest validation accuracy

# Define hyperparameters
task='answer-prediction'
n_classes = 595
alpha = 0.9
batch_size = 256
lr = 0.01
n_heads = [2,4]
feedforward_dim = 1024

# Set up DataLoaders
train_loader = DataLoader(CLIP_training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CLIP_validation_set, batch_size=batch_size, shuffle=False)

# Run experiments
for nhead in n_heads:
    # Define model, loss function, and optimizer
    model = LightweightModel(
        nhead=nhead,
        feedforward_dim=feedforward_dim,
        dropout_rate=0.1,
        n_classes=n_classes
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=alpha)

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_set=CLIP_validation_set,
        eval_batch_size=len(CLIP_validation_set),
        loss_fn=loss_fn,
        optimizer=optimizer,
        n_epochs=9,
        category_id2name=category_id2name,
        early_stopping=False,
        eval_every=1,
        epoch_eval_every=3,
        wandb_config={
            "lr":lr,
            "alpha":alpha,
            "nheads": nhead,
            "feedforward_dim":feedforward_dim
        },
        wandb_tags=['prototyping', task, "CLIP"],
        task=task
    )

#------------- Model Evaluation: Binary Classification Task -------------#
# Load the state of the saved model
model_state_dict = torch.load(f"{ROOT_DATA_PATH}/artifacts/challenge1_model/baseline.pt", map_location=device)

# Load the model and freeze parameters
model = VQAModel(
    nhead=1,
    n_classes=1
).to(device)
model.load_state_dict(model_state_dict, )
model.eval()

# Set up DataLoader for batched operation
subset = [testing_set[idx] for idx in range(100, 200)]
test_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
samples = next(iter(test_loader))

# Get predictions
preds = test_model_binary(model=model, samples=samples)

# Save predictions
save_predictions(preds, root_path=ROOT_DATA_PATH, filename="rob_stallman_challenge1.pkl")

#------------- Model Evaluation: Answer Prediction Task -------------#
# Load the state of the saved model
model_state_dict = torch.load(f"{ROOT_DATA_PATH}/artifacts/challenge2_model/baseline.pt", map_location=device)

# Load the model and freeze parameters
model = VQAModel(
    image_droput_rate=0.2,
    embed_dim=128,
    nhead=4,
    fusion_dim=512,
    n_classes=595,
).to(device)
model.load_state_dict(model_state_dict)
model.eval()

# Set up DataLoaders
subset = [testing_set[idx] for idx in range(100, 200)]
test_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
samples = next(iter(test_loader))

# Get predictions
answers = test_model_answer(model=model, samples=samples, category_id2name=category_id2name)

# Get image URLs
with open(ANNOTATION_PATH_TEST, 'r') as f:
    test_annotations = json.load(f)
test_image_urls = [test_annotations[idx]['image'] for idx in range(100,200)]

# Construct prediction dictionary
preds = [{"image":test_image_urls[idx], "answer":answers[idx]} for idx in range(len(answers))]

# Save predictions
save_predictions(preds, root_path=ROOT_DATA_PATH, filename="rob_stallman_challenge2.json")

#------------- Model Evaluation: Binary Classification Task -------------#
# Load the state of the saved model
model_state_dict = torch.load(f"{ROOT_DATA_PATH}/artifacts/challenge3_model/baseline.pt", map_location=device)

# Load the model and freeze parameters
# Using sweet-planet-39
model = LightweightModel(
    nhead=1,
    feedforward_dim=512,
    dropout_rate=0.1,
    n_classes=1
).to(device)
model.load_state_dict(model_state_dict, )
model.eval()

# Set up DataLoader for batched operation
subset = [CLIP_testing_set[idx] for idx in range(100, 200)]
test_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
samples = next(iter(test_loader))

# Get predictions
preds = test_model_binary(model=model, samples=samples)

# Save predictions
save_predictions(preds, root_path=ROOT_DATA_PATH, filename="rob_stallman_challenge3.pkl")

#------------- Model Evaluation: Answer Prediction Task -------------#
# Load the state of the saved model
model_state_dict = torch.load(f"{ROOT_DATA_PATH}/artifacts/challenge4_model/baseline.pt", map_location=device)

# Load the model and freeze parameters
# Using jolly-breeze-40
model = LightweightModel(
    nhead=2,
    feedforward_dim=1024,
    dropout_rate=0.1,
    n_classes=595
).to(device)
model.load_state_dict(model_state_dict, )
model.eval()

# Set up DataLoader for batched operation
subset = [CLIP_testing_set[idx] for idx in range(100, 200)]
test_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
samples = next(iter(test_loader))

# Get predictions
answers = test_model_answer(model=model, samples=samples, category_id2name=category_id2name)

# Get image URLs
with open(ANNOTATION_PATH_TEST, 'r') as f:
    test_annotations = json.load(f)
test_image_urls = [test_annotations[idx]['image'] for idx in range(100,200)]

# Construct prediction dictionary
preds = [{"image":test_image_urls[idx], "answer":answers[idx]} for idx in range(len(answers))]

# Save predictions
save_predictions(preds, root_path=ROOT_DATA_PATH, filename="rob_stallman_challenge4.json")

VIZWIZ_MEAN_100 = [0.489218533039093, 0.4372936487197876, 0.38480180501937866]
VIZWIZ_STD_100 = [0.2866476774215698, 0.27600425481796265, 0.27667489647865295]
VIZWIZ_MEAN_075 = [0.48784229159355164, 0.4363100230693817, 0.38287216424942017]
VIZWIZ_STD_075 = [0.28680506348609924, 0.27655652165412903, 0.27689728140830994]

new_dir = os.path.join(ROOT_DATA_PATH, "100_data")
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

NEW_TOK2IDX_PATH = os.path.join(new_dir, "tok2idx.json")
NEW_IDX2TOK_PATH = os.path.join(new_dir, "idx2tok.json")
NEW_VOCAB_COUNTER_PATH = os.path.join(new_dir, "vocab_counter.json")
NEW_EMBEDDED_QUESTIONS_PATH_TRAIN = os.path.join(new_dir, "questions_train.pt")
NEW_EMBEDDED_QUESTIONS_PATH_VAL = os.path.join(new_dir, "questions_val.pt")
NEW_NAME2ID_PATH = os.path.join(new_dir, "name2id.json")
NEW_ID2NAME_PATH = os.path.join(new_dir, "id2name.json")
NEW_ENCODED_ANSWERS_PATH_TRAIN = os.path.join(new_dir, "answers_train.pt")
NEW_ENCODED_ANSWERS_PATH_VAL = os.path.join(new_dir, "answers_val.pt")


# Repeat preprocessing
new_train_set, new_val_set, _ = load_datasets(mean=VIZWIZ_MEAN_100, std=VIZWIZ_STD_100, fDataPercentage=1.0)
print("Unembedded datasets are loaded with lengths:")
print(f"Training: {len(new_train_set)}")
print(f"Validation: {len(new_val_set)}")

### Process questions

# Look for previously-embedded questions
try:
    questions_train = torch.load(NEW_EMBEDDED_QUESTIONS_PATH_TRAIN)
    questions_train = questions_train.to(dtype=torch.long)
    questions_val = torch.load(NEW_EMBEDDED_QUESTIONS_PATH_VAL)
    questions_val = questions_val.to(dtype=torch.long)

    with open(NEW_TOK2IDX_PATH, 'r') as f:
        tok2idx = json.load(f)
    with open(NEW_IDX2TOK_PATH, 'r') as f:
        idx2tok = json.load(f)
        idx2tok = {int(k) if k.isdigit() else k:v for k,v in idx2tok.items()}
    with open(NEW_VOCAB_COUNTER_PATH, 'r') as f:
        counter_dict = json.load(f)
        counter = Counter(counter_dict)  
except FileNotFoundError:
    print("Processing questions from scratch...")
    # Tokenize the train, val, and test datasets
    train_tokens = batch_tokenize(new_train_set, nlp)
    print(f"Tokenized {len(train_tokens)} training samples")
    val_tokens = batch_tokenize(new_val_set, nlp)
    print(f"Tokenized {len(val_tokens)} validation samples")

    # Build vocabulary from the training set
    tok2idx, idx2tok, counter = build_vocab(train_tokens, max_vocab_size=1110)
    print(f"Vocab created with {len(tok2idx)} entries")
    
    # Save vocab
    with open(NEW_TOK2IDX_PATH, 'w') as f:
        json.dump(tok2idx, f, indent=4)
    with open(NEW_IDX2TOK_PATH, 'w') as f:
        json.dump(idx2tok, f, indent=4)
    with open(NEW_VOCAB_COUNTER_PATH, 'w') as f:
        json.dump(counter, f, indent=4)

    # Encode the train, val, and test questions
    questions_train = torch.tensor(embed_dataset(train_tokens, tok2idx), dtype=torch.long)
    questions_val = torch.tensor(embed_dataset(val_tokens, tok2idx), dtype=torch.long)

    # Save embedded questions
    torch.save(questions_train, NEW_EMBEDDED_QUESTIONS_PATH_TRAIN)
    torch.save(questions_val, NEW_EMBEDDED_QUESTIONS_PATH_VAL)

print("Training questions embedded with shape", questions_train.shape)
print("Validation questions embedded with shape", questions_val.shape)
print("Test questions embedded with shape", questions_test.shape)

### Process answers

# Look for previously-embedded answers
try:
    new_answers_train = torch.load(NEW_ENCODED_ANSWERS_PATH_TRAIN)
    new_answers_val = torch.load(NEW_ENCODED_ANSWERS_PATH_VAL)

    with open(NEW_NAME2ID_PATH, 'r') as f:
        category_name2id = json.load(f)
    with open(NEW_ID2NAME_PATH, 'r') as f:
        category_id2name = json.load(f)
        category_id2name = {int(k) if k.isdigit() else k:v for k,v in category_id2name.items()}
except FileNotFoundError:
    print("Processing answers from scratch...")
    # Generate a list of qualified answers
    all_qualified_answers = []
    for idx, training_sample in enumerate(new_train_set):
        if (idx % 50) == 49:
            print(f"Working on sample: {idx+1}/{new_train_set.iN}")
        sample_answers = get_sample_corpus_entries(training_sample)
        if sample_answers:
            all_qualified_answers += sample_answers

    # Take all answers meeting the minimum frequency cutoff
    qualified_answer_counts = Counter(all_qualified_answers)
    answer_corpus = [ans for ans, counts in qualified_answer_counts.most_common() if counts >= FREQUENCY_CUTOFF]
    n_answers = len(answer_corpus)
    print(f"Answer corpus contains {n_answers} real answers")

    # Create categories for answer corpus
    category_name2id = {ans:idx for idx, ans in enumerate(answer_corpus)}
    category_id2name ={idx:ans for idx, ans in enumerate(answer_corpus)}
    category_id2name[n_answers] = 'other_categories'        # Bucket for answers not in the corpus
    category_name2id['other_categories'] = n_answers

    # Save categories
    with open(NEW_NAME2ID_PATH, 'w') as f:
        json.dump(category_name2id, f)
    with open(NEW_ID2NAME_PATH, 'w') as f:
        json.dump(category_id2name, f)

    # Encode the train and val answers (no test answers in dataset)
    new_answers_train = torch.stack([encode_sample(sample, category_name2id) for sample in new_train_set], dim=0)
    new_answers_val = torch.stack([encode_sample(sample, category_name2id) for sample in new_val_set], dim=0)
    
    # Save encoded answers
    torch.save(new_answers_train, NEW_ENCODED_ANSWERS_PATH_TRAIN)
    torch.save(new_answers_val, NEW_ENCODED_ANSWERS_PATH_VAL)

print("Training answers encoded as categories with shape", new_answers_train.shape)
print("Validation answers encoded as categories with shape", new_answers_val.shape)

### Build datasets with embedded questions and answers
tfms = v2.Compose([
        v2.Normalize(VIZWIZ_MEAN_100, VIZWIZ_STD_100)
    ])
training_set_expanded = VizWizLoader(
    strFolder=TRAIN_PATH,
    strAnnotationPath=ANNOTATION_PATH_TRAIN,
    fDataPercentage=1.0,
    tTransform=tfms,
    embedded_questions=questions_train,
    embedded_answers=new_answers_train
    )
validation_set_expanded = VizWizLoader(
    strFolder=VAL_PATH,
    strAnnotationPath=ANNOTATION_PATH_VAL,
    fDataPercentage=1.0,
    tTransform=tfms,
    embedded_questions=questions_val,
    embedded_answers=new_answers_val
    )
print("Finished preprocessing steps")

#------------- Custom Model Training: Binary Classification Task -------------#
# Define hyperparameters
task='binary-prediction'
batch_size=1024
lr=0.01
alpha=0.9
nhead=1
n_classes=1

# Load the state of the saved model
model_state_dict = torch.load(f"{ROOT_DATA_PATH}/artifacts/challenge1_model/baseline.pt", map_location=device)
print("Loaded model state dictionary")

# Set up DataLoaders
train_loader = DataLoader(training_set_expanded, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set_expanded, batch_size=batch_size, shuffle=False)

# Load the model and freeze parameters
model = VQAModel(
    nhead=nhead,
    n_classes=n_classes
).to(device)
model.load_state_dict(model_state_dict)
print("Loaded model from pre-saved states")

# Define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=alpha)

# Train model
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    val_set=validation_set,
    eval_batch_size=len(validation_set),
    loss_fn=loss_fn,
    optimizer=optimizer,
    n_epochs=18,
    category_id2name=category_id2name,
    early_stopping=True,
    patience=6,
    min_delta=0.01,
    eval_every=1,
    epoch_eval_every=3,
    wandb_config={
        "lr":lr,
        "alpha":alpha,
        "nheads": nhead,
        "fDataPercentage":1.0
    },
    wandb_tags=['data-efficiency', task],
    task=task
)
# Load pre-trained features
CLIP_image_train = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_train_CLIP_Image.pkl"))
CLIP_image_val = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_val_CLIP_Image.pkl"))
CLIP_image_test = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_test_CLIP_Image.pkl"))
CLIP_text_train = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_train_CLIP_Text.pkl"))
CLIP_text_val = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_val_CLIP_Text.pkl"))
CLIP_text_test = torch.load(os.path.join(ROOT_DATA_PATH, "VizWiz_CLIP", "VizWiz_test_CLIP_Text.pkl"))

# Limit dataset size
#num_samples = 15392     # TODO: Change me! (20523)
CLIP_image_train_expanded = CLIP_image_train
CLIP_text_train_expanded = CLIP_text_train

# Load pre-trained datasets
CLIP_training_set_expanded = CLIP_dataset(
    strAnnotationPath=ANNOTATION_PATH_TRAIN,
    embedded_images=CLIP_image_train_expanded,
    embedded_questions=CLIP_text_train_expanded,
    embedded_answers=new_answers_train
)
CLIP_validation_set_expanded = CLIP_dataset(
    strAnnotationPath=ANNOTATION_PATH_VAL,
    embedded_images=CLIP_image_val,
    embedded_questions=CLIP_text_val,
    embedded_answers=new_answers_val
)
print("Finished loading pre-processed features")

#------------- Pretrained Model Training: Binary Classification Task -------------#
# Define hyperparameters
task='binary-prediction'
batch_size=1024
lr=0.01
alpha=0.9
nhead=1
feedforward_dim=512
dropout_rate=0.1
n_classes=1

# Load the state of the saved model
model_state_dict = torch.load(f"{ROOT_DATA_PATH}/artifacts/challenge3_model/baseline.pt", map_location=device)
print("Loaded model state dictionary")

# Set up DataLoaders
train_loader = DataLoader(CLIP_training_set_expanded, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CLIP_validation_set_expanded, batch_size=batch_size, shuffle=False)

# Load the model and freeze parameters
# Using sweet-planet-39
model = LightweightModel(
    nhead=nhead,
    feedforward_dim=feedforward_dim,
    dropout_rate=dropout_rate,
    n_classes=n_classes
).to(device)
model.load_state_dict(model_state_dict)

# Define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=alpha)

# Train model
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    val_set=CLIP_validation_set,
    eval_batch_size=len(CLIP_validation_set),
    loss_fn=loss_fn,
    optimizer=optimizer,
    n_epochs=9,
    category_id2name=category_id2name,
    early_stopping=False,
    patience=6,
    min_delta=0.01,
    eval_every=1,
    epoch_eval_every=3,
    wandb_config={
        "lr":lr,
        "alpha":alpha,
        "nheads": nhead,
        "feedforward_dim":feedforward_dim,
        "dropout_rate":dropout_rate,
        "fDataPercentage":1.0
    },
    wandb_tags=['data-efficiency', task],
    task=task
)
