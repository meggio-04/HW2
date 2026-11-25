import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from funzioni import *
import numpy as np
from pathlib import Path
import pandas as pd
import time

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print(f"Using device: {device}\n")

""" 
    PREPROCESSING
"""

# Pesi pre-addestrati
weights = models.MobileNet_V3_Small_Weights.DEFAULT

# Preprocessing ufficiale per validation e test
eval_transform = weights.transforms()

# Statistiche di normalizzazione dei pesi
MEAN = eval_transform.mean
STD = eval_transform.std

# Augmentation per il training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


""" 
    DATASET
"""

DATA_ROOT = Path("./data")

# Dataset trainval con augmentation
trainval_aug = datasets.OxfordIIITPet( root=DATA_ROOT, split="trainval", target_types="category", transform=train_transform, download=True)

# Dataset trainval senza augmentation
trainval_eval = datasets.OxfordIIITPet( root=DATA_ROOT, split="trainval", target_types="category", transform=eval_transform, download=True)

# Dataset test senza augmentation
test_dataset_full = datasets.OxfordIIITPet( root=DATA_ROOT, split="test", target_types="category", transform=eval_transform, download=True)

# class info##################################

NUM_CLASSES = 37
CLASS_NAMES = trainval_eval.classes

trainval_labels = np.array(trainval_eval._labels)
test_labels = np.array(test_dataset_full._labels)

print("Trainval images:", len(trainval_eval))
print("Test images:", len(test_dataset_full))
print("Number of classes:", len(CLASS_NAMES))

###############################################

# train/validation split ######################

indices = np.arange(len(trainval_eval))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=trainval_labels
)
###############################################


# Applichiamo gli stessi indici dello split
train_dataset = Subset(trainval_aug, train_idx)
train_eval_dataset = Subset(trainval_eval, train_idx)
val_dataset = Subset(trainval_eval, val_idx)
test_dataset = test_dataset_full


""" 
    CHECK SPLIT DISTRIBUTION -- CLASS DISTRIBUTION
"""

counts = pd.DataFrame({
    "Breed": CLASS_NAMES,
    "Train": np.bincount(
        trainval_labels[train_idx], minlength=NUM_CLASSES
    ),
    "Validation": np.bincount(
        trainval_labels[val_idx], minlength=NUM_CLASSES
    ),
    "Test": np.bincount(
        test_labels, minlength=NUM_CLASSES
    )
})

print("\nClass distribution:")
print(counts.to_string(index=False))

print("\nTotal images:")
print("Train:", counts["Train"].sum())
print("Validation:", counts["Validation"].sum())
print("Test:", counts["Test"].sum())

# Save class distribution
Path("./results").mkdir(parents=True, exist_ok=True)

counts.to_csv(
    "./results/class_counts.csv",
    index=False
)

""" 
    DATALOADER
"""

BATCH_SIZE = 32

# Generatore con seed fisso per il training
train_generator = torch.Generator()
train_generator.manual_seed(SEED)

# Training: augmentation + shuffle
train_loader = DataLoader( train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator)

# Training senza augmentation: per valutare le prestazioni
train_eval_loader = DataLoader( train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False )

# Validation: nessuno shuffle
val_loader = DataLoader( val_dataset, batch_size=BATCH_SIZE, shuffle=False )

# Test: nessuno shuffle
test_loader = DataLoader( test_dataset, batch_size=BATCH_SIZE, shuffle=False,)


""" 
    METHODS 1 : Full fine-tuning
    Replace the final classification layer and train the entire network, including its convolutional layers.
    - all parameters are trainable
"""

# Load pre-trained MobileNetV3-Small
model_full = models.mobilenet_v3_small(weights=weights)

# Replace the final classification layer, 3 or -1
# NB : classifier[0]: Linear (576 → 1024), classifier[1]: Hardswish, classifier[2]: Dropout, classifier[3]: Linear (1024 → 1000)
num_features = model_full.classifier[-1].in_features

model_full.classifier[-1] = nn.Linear( num_features, NUM_CLASSES)

# move complete model to the device
model_full = model_full.to(device)

for param in model_full.parameters():
    param.requires_grad = True

# Report
# Count total and trainable parameters
total_params = sum( p.numel() for p in model_full.parameters())

trainable_params_full = sum( p.numel() for p in model_full.parameters() if p.requires_grad)

print("\nFull Fine-Tuning:")
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params_full)

print("\nClassifier:")
print(model_full.classifier)

"""
    TRAINING: FULL FINE-TUNING
"""

NUM_EPOCHS = 15
LEARNING_RATE_FULL = 1e-4

# Loss function
criterion_full = nn.CrossEntropyLoss()

# Optimizer
optimizer_full = optim.Adam( model_full.parameters(), lr=LEARNING_RATE_FULL)

# Variables for checkpoint selection
best_val_f1_full = -1
best_epoch_full = 0

# Reset seed for reproducible shuffle
train_generator.manual_seed(SEED)

# Start training time
if device.type == "cuda":
    torch.cuda.synchronize()

best_val_accuracy_full = None

start_time = time.perf_counter()

for epoch in range(NUM_EPOCHS):
    model_full.train()
    running_loss = 0.0

    # Training
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer_full.zero_grad()

        outputs = model_full(images)
        loss = criterion_full(outputs, labels)

        loss.backward()
        optimizer_full.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model_full.eval()

    val_loss_total = 0.0
    val_true = []
    val_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model_full(images)
            loss = criterion_full(outputs, labels)

            val_loss_total += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            val_true.extend(labels.cpu().numpy())
            val_predictions.extend(predictions.cpu().numpy())

    val_loss = val_loss_total / len(val_loader.dataset)
    val_accuracy = accuracy_score(val_true, val_predictions)

    val_f1 = f1_score( val_true,  val_predictions,  average="macro")

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f} "
        f"Val Macro F1: {val_f1:.4f}"
    )

    # Save the best checkpoint based on validation Macro F1
    if val_f1 > best_val_f1_full:

        best_val_f1_full = val_f1
        best_epoch_full = epoch + 1

        best_val_accuracy_full = val_accuracy

        torch.save({
            "model_state_dict": model_full.state_dict(),
            "optimizer_state_dict": optimizer_full.state_dict(),
            "epoch": int(best_epoch_full),
            "val_macro_f1": float(best_val_f1_full),
            "val_accuracy": float(best_val_accuracy_full),
            "learning_rate": float(LEARNING_RATE_FULL),
            "batch_size": int(BATCH_SIZE),
            "seed": int(SEED)
        }, "./results/best_model_full.pth")

# End training time
if device.type == "cuda":
    torch.cuda.synchronize()

full_training_time = time.perf_counter() - start_time

# Load the best checkpoint
checkpoint = torch.load(
    "./results/best_model_full.pth",
    map_location=device,
    weights_only=True
)

model_full.load_state_dict(checkpoint["model_state_dict"])

print("\nFull Fine-Tuning - Best Checkpoint:")
print("Best Epoch:", best_epoch_full)
print("Best Validation Macro F1:", best_val_f1_full)
print(f"Training Time: {full_training_time:.2f} seconds")

""" FINAL EVALUATION - FULL FINE-TUNING """

# Training evaluation (without augmentation)
full_train_results = evaluate_model(
    model_full,
    train_eval_loader,
    device,
    NUM_CLASSES
)

# Test evaluation
full_test_results = evaluate_model(
    model_full,
    test_loader,
    device,
    NUM_CLASSES
)


# Print results
print("\nFULL FINE-TUNING - FINAL RESULTS")

print("\nTraining Performance:")
print(f"Training Accuracy: {full_train_results['accuracy']:.4f}")
print(f"Training Macro F1: {full_train_results['macro_f1']:.4f}")

print("\nTest Performance:")
print(f"Test Accuracy: {full_test_results['accuracy']:.4f}")
print(f"Test Macro F1: {full_test_results['macro_f1']:.4f}")



"""
    METHOD 2: CLASSIFIER-HEAD TRAINING
    Freeze the convolutional feature extractor and train
    the fully connected classification layers.

    NB : 
    Feature Extractor                   -->     frozen
    classifier[0] Linear (576 → 1024)	-->     Trainable
    classifier[1] Hardswish	            -->     Non ha parametri
    classifier[2] Dropout	            -->     Non ha parametri
    classifier[3] Linear (1024 → 37)	-->     Trainable

"""
# Reset seeds for reproducibility
set_seed(SEED)
train_generator.manual_seed(SEED)

# Load a fresh pre-trained MobileNetV3-Small
model_head = models.mobilenet_v3_small(weights=weights)

# Replace the final classification layer (1000 -> 37)
num_features = model_head.classifier[-1].in_features

model_head.classifier[-1] = nn.Linear(
    num_features,
    NUM_CLASSES
)

# Freeze the convolutional feature extractor
for param in model_head.features.parameters():
    param.requires_grad = False

# Unfreeze the fully connected classification layers
for param in model_head.classifier.parameters():
    param.requires_grad = True

# Move model to device
model_head = model_head.to(device)

# Report
# Count total and trainable parameters
total_params_head = sum(
    p.numel() for p in model_head.parameters()
)

trainable_params_head = sum(
    p.numel() for p in model_head.parameters()
    if p.requires_grad
)

print("\nClassifier-Head Training:")
print("Total parameters:", total_params_head)
print("Trainable parameters:", trainable_params_head)

print("\nClassifier:")
print(model_head.classifier)

"""
    TRAINING: CLASSIFIER-HEAD TRAINING
"""

NUM_EPOCHS_HEAD = 15
LEARNING_RATE_HEAD = 1e-4

# Loss function
criterion_head = nn.CrossEntropyLoss()

# Optimizer: only trainable parameters
optimizer_head = optim.Adam( filter(lambda p: p.requires_grad, model_head.parameters()),  lr=LEARNING_RATE_HEAD )

# Variables for checkpoint selection
best_val_f1_head = -1
best_epoch_head = 0

best_val_accuracy_head = None

# Start training time
if device.type == "cuda":
    torch.cuda.synchronize()

start_time = time.perf_counter()

for epoch in range(NUM_EPOCHS_HEAD):

    model_head.train()

    # Keep convolutional feature extractor frozen
    model_head.features.eval()
    model_head.classifier.train()

    running_loss = 0.0

    # Training
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer_head.zero_grad()

        outputs = model_head(images)
        loss = criterion_head(outputs, labels)

        loss.backward()
        optimizer_head.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model_head.eval()

    val_loss_total = 0.0
    val_true = []
    val_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model_head(images)
            loss = criterion_head(outputs, labels)

            val_loss_total += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            val_true.extend(labels.cpu().numpy())
            val_predictions.extend(predictions.cpu().numpy())

    val_loss = val_loss_total / len(val_loader.dataset)

    val_accuracy = accuracy_score(val_true, val_predictions)

    val_f1 = f1_score( val_true, val_predictions, average="macro")

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS_HEAD}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f} "
        f"Val Macro F1: {val_f1:.4f}"
    )

    # Save the best checkpoint based on validation Macro F1
    if val_f1 > best_val_f1_head:

        best_val_f1_head = val_f1
        best_epoch_head = epoch + 1

        best_val_accuracy_head = val_accuracy

        torch.save({
            "model_state_dict": model_head.state_dict(),
            "optimizer_state_dict": optimizer_head.state_dict(),
            "epoch": int(best_epoch_head),
            "val_macro_f1": float(best_val_f1_head),
            "val_accuracy": float(best_val_accuracy_head),
            "learning_rate": float(LEARNING_RATE_HEAD),
            "batch_size": int(BATCH_SIZE),
            "seed": int(SEED)
        }, "./results/best_model_head.pth")


# End training time
if device.type == "cuda":
    torch.cuda.synchronize()

head_training_time = time.perf_counter() - start_time

# Load the best checkpoint
checkpoint = torch.load(
    "./results/best_model_head.pth",
    map_location=device,
    weights_only=True
)

model_head.load_state_dict(checkpoint["model_state_dict"])

print("\nClassifier-Head Training - Best Checkpoint:")
print("Best Epoch:", best_epoch_head)
print("Best Validation Macro F1:", best_val_f1_head)
print(f"Training Time: {head_training_time:.2f} seconds")

""" FINAL EVALUATION - CLASSIFIER-HEAD """

# Training evaluation (without augmentation)
head_train_results = evaluate_model(
    model_head,
    train_eval_loader,
    device,
    NUM_CLASSES
)

# Test evaluation
head_test_results = evaluate_model(
    model_head,
    test_loader,
    device,
    NUM_CLASSES
)


# Print results
print("\nCLASSIFIER-HEAD - FINAL RESULTS")

print("\nTraining Performance:")
print(f"Training Accuracy: {head_train_results['accuracy']:.4f}")
print(f"Training Macro F1: {head_train_results['macro_f1']:.4f}")

print("\nTest Performance:")
print(f"Test Accuracy: {head_test_results['accuracy']:.4f}")
print(f"Test Macro F1: {head_test_results['macro_f1']:.4f}")


""" 
    METHOD 3 : FEATURE EXTRACTION + LINEAR SVM
    Feature extraction with a traditional classifier
    freeze the pre-trained network and extract feature vectors from the output of its feature extractor, 
    before the classifier (after the network’s pooling operation).
    Train a classifier of your choice from scikit-learn on the training features and evaluate it on the test features. 
    Examples include a linear SVM, logistic regression, or random forest. 
    Optional: compare more than one traditional classifier.

"""

# Load pre-trained MobileNetV3-Small
model_feex = models.mobilenet_v3_small(weights=weights).to(device)

# Freeze all network parameters
for param in model_feex.parameters():
    param.requires_grad = False

# Remove classifier and keep features + average pooling
feature_extractor = nn.Sequential(*list(model_feex.children())[:-1]).to(device)

feature_extractor.eval()

# Start timing
if device.type == "cuda":
    torch.cuda.synchronize()

start_time = time.perf_counter()

# Extract training and validation features
train_features, train_feature_labels = extract_features( train_eval_loader, feature_extractor, device)

val_features, val_feature_labels = extract_features( val_loader, feature_extractor, device)

# Train SVM and select best C using validation Macro F1
best_model, best_C, best_val_accuracy, best_val_f1 = select_best_svm(
    train_features,
    train_feature_labels,
    val_features,
    val_feature_labels
)

if device.type == "cuda":
    torch.cuda.synchronize()

method3_time = time.perf_counter() - start_time

print("\nBest C:", best_C)
print("Best Validation Macro F1:", best_val_f1)
print(f"Method 3 time: {method3_time:.2f} seconds")


# Extract test features after model selection
test_features, test_feature_labels = extract_features( test_loader, feature_extractor, device)

# Final evaluation on test set
test_predictions = best_model.predict(test_features)

test_accuracy = accuracy_score(test_feature_labels, test_predictions)
test_f1 = f1_score( test_feature_labels, test_predictions, average="macro")

# Calculate SVM confusion matrix
cm_svm = confusion_matrix(
    test_feature_labels,
    test_predictions,
    labels=np.arange(NUM_CLASSES)
)

# Count learned parameters
cnn_trainable_params = sum(  p.numel() for p in model_feex.parameters()  if p.requires_grad  )

svm_classifier = best_model.named_steps["classifier"]

svm_parameters = ( svm_classifier.coef_.size +  svm_classifier.intercept_.size )

print("\nLinear SVM - Test Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_f1:.4f}")
print("MobileNet trainable parameters:", cnn_trainable_params)
print("SVM learned parameters:", svm_parameters)

# Training predictions
train_predictions_svm = best_model.predict(train_features)

train_accuracy_svm = accuracy_score( train_feature_labels, train_predictions_svm)

train_f1_svm = f1_score( train_feature_labels, train_predictions_svm, average="macro")

""" 
    CONFUSION MATRIX - ALL THREE METHODS
"""

# METHOD 1: FULL FINE-TUNING
save_confusion_matrix(
    full_test_results["confusion_matrix"],
    CLASS_NAMES,
    "Full Fine-Tuning",
    "./results/confusion_matrix_full.png"
)

# METHOD 2: CLASSIFIER-HEAD
save_confusion_matrix(
    head_test_results["confusion_matrix"],
    CLASS_NAMES,
    "Classifier-Head",
    "./results/confusion_matrix_head.png"
)

# METHOD 3: FEATURE EXTRACTION + SVM
save_confusion_matrix(
    cm_svm,
    CLASS_NAMES,
    "Feature Extraction + Linear SVM",
    "./results/confusion_matrix_svm.png"
)

# TOP CONFUSIONS - FULL FINE-TUNING

top_full = get_top_confusions(
    full_test_results["confusion_matrix"],
    CLASS_NAMES,
    top_k=10
)

# TOP CONFUSIONS - CLASSIFIER HEAD

top_head = get_top_confusions(
    head_test_results["confusion_matrix"],
    CLASS_NAMES,
    top_k=10
)

# TOP CONFUSIONS - FEATURE EXTRACTION + SVM

top_svm = get_top_confusions(
    cm_svm,
    CLASS_NAMES,
    top_k=10
)

print("\nTOP CONFUSIONS - FULL FINE-TUNING")
print(top_full.to_string(index=False))

print("\nTOP CONFUSIONS - CLASSIFIER HEAD")
print(top_head.to_string(index=False))

print("\nTOP CONFUSIONS - FEATURE EXTRACTION + SVM")
print(top_svm.to_string(index=False))

"""
    SAVE PREDICTION EXAMPLES - ALL THREE METHODS
"""

# METHOD 1: FULL FINE-TUNING
save_prediction_examples(
    dataset=test_dataset,
    y_true=full_test_results["y_true"],
    y_pred=full_test_results["y_pred"],
    class_names=CLASS_NAMES,
    mean=MEAN,
    std=STD,
    output_path="./results/full_prediction_examples.png"
)


# METHOD 2: CLASSIFIER-HEAD
save_prediction_examples(
    dataset=test_dataset,
    y_true=head_test_results["y_true"],
    y_pred=head_test_results["y_pred"],
    class_names=CLASS_NAMES,
    mean=MEAN,
    std=STD,
    output_path="./results/head_prediction_examples.png"
)


# METHOD 3: FEATURE EXTRACTION + SVM
save_prediction_examples(
    dataset=test_dataset,
    y_true=test_feature_labels,
    y_pred=test_predictions,
    class_names=CLASS_NAMES,
    mean=MEAN,
    std=STD,
    output_path="./results/svm_prediction_examples.png"
)

"""
    FINAL COMPARISON - ALL THREE METHODS
"""

evaluation_results = pd.DataFrame([

    # METHOD 1: FULL FINE-TUNING
    {
        "Method": "Full Fine-Tuning",

        "Train_Accuracy": full_train_results["accuracy"],
        "Train_Macro_F1": full_train_results["macro_f1"],

        "Validation_Accuracy": best_val_accuracy_full,
        "Validation_Macro_F1": best_val_f1_full,

        "Test_Accuracy": full_test_results["accuracy"],
        "Test_Macro_F1": full_test_results["macro_f1"],

        "Best_Epoch": best_epoch_full,
        "Best_C": np.nan,

        "Trainable_Parameters": trainable_params_full,
        "SVM_Learned_Parameters": np.nan,

        "Training_Time_seconds": full_training_time
    },

    # METHOD 2: CLASSIFIER-HEAD
    {
        "Method": "Classifier-Head",

        "Train_Accuracy": head_train_results["accuracy"],
        "Train_Macro_F1": head_train_results["macro_f1"],

        "Validation_Accuracy": best_val_accuracy_head,
        "Validation_Macro_F1": best_val_f1_head,

        "Test_Accuracy": head_test_results["accuracy"],
        "Test_Macro_F1": head_test_results["macro_f1"],

        "Best_Epoch": best_epoch_head,
        "Best_C": np.nan,

        "Trainable_Parameters": trainable_params_head,
        "SVM_Learned_Parameters": np.nan,

        "Training_Time_seconds": head_training_time
    },

    # METHOD 3: FEATURE EXTRACTION + SVM
    {
        "Method": "Feature Extraction + Linear SVM",

        "Train_Accuracy": train_accuracy_svm,
        "Train_Macro_F1": train_f1_svm,

        "Validation_Accuracy": best_val_accuracy,
        "Validation_Macro_F1": best_val_f1,

        "Test_Accuracy": test_accuracy,
        "Test_Macro_F1": test_f1,

        "Best_Epoch": np.nan,
        "Best_C": best_C,

        "Trainable_Parameters": cnn_trainable_params,
        "SVM_Learned_Parameters": svm_parameters,

        "Training_Time_seconds": method3_time
    }

])


# Display comparative results
print("\nFINAL COMPARISON - ALL METHODS")

print(evaluation_results.to_string(index=False))


# Save final comparison
evaluation_results.to_csv(
    "./results/final_comparison.csv",
    index=False
)