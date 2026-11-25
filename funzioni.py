import torch
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from sklearn.exceptions import ConvergenceWarning





""" REPRODUCIBILITY """

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Operazioni deterministiche, quando disponibili
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

        # Enable deterministic algorithms (warning mode)
    torch.use_deterministic_algorithms(
        True,
        warn_only=True)


""" EXTRACT_FEATURES """

# Function to extract features from a DataLoader
def extract_features(loader, feature_extractor, device):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Extract features
            raw_features = feature_extractor(images)
            # Flatten feature vectors
            features = torch.flatten(raw_features, 1)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0)
    )


""" BEST SVM """
def select_best_svm(X_train, y_train, X_val, y_val):

    best_model = None
    best_C = None
    best_val_f1 = -1
    best_val_accuracy = None

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:

        print(f"\nStarting SVM with C={C}", flush=True)

        start = time.perf_counter()

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                C=C,
                random_state=SEED,
                max_iter=10000, # 10.000
                dual=False,
                tol=1e-3
            ))
        ])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)

            model.fit(X_train, y_train)

        elapsed = time.perf_counter() - start

        # Check convergence warnings
        convergence_warning = any(
            issubclass(w.category, ConvergenceWarning)
            for w in caught
        )

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score( y_val, predictions, average="macro")

        print(
            f"C={C}: Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}, "
            f"Time={elapsed:.2f}s, "
            f"Convergence warning={convergence_warning}",
            flush=True
        )

        if macro_f1 > best_val_f1:

            best_model = model
            best_C = C
            best_val_f1 = macro_f1
            best_val_accuracy = accuracy

    return ( best_model, best_C, best_val_accuracy, best_val_f1 )



""" SAVE_PREDICTION """

def save_prediction_examples( dataset, y_true, y_pred, class_names, mean, std, output_path, seed=SEED):

    # Reproducible random selection
    rng = np.random.default_rng(seed)

    # Find correct and incorrect predictions
    correct = np.where(y_true == y_pred)[0]
    incorrect = np.where(y_true != y_pred)[0]

    # Randomly select up to 4 examples for each group
    groups = [ (rng.choice(correct, min(4, len(correct)), replace=False), "Correct"),
                (rng.choice(incorrect, min(4, len(incorrect)), replace=False), "Incorrect")
            ]

    # Convert normalization statistics into tensors
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row, (indices, title) in enumerate(groups):
        for col in range(4):
            axes[row, col].axis("off")

            if col < len(indices):
                idx = indices[col]
                # Get image
                image, _ = dataset[idx]
                # Reverse normalization
                image = (image * std + mean).clamp(0, 1)
                image = image.permute(1, 2, 0).numpy()

                # Get true and predicted class names
                true = class_names[int(y_true[idx])]
                pred = class_names[int(y_pred[idx])]

                # Display image and predictions
                axes[row, col].imshow(image)

                axes[row, col].set_title(
                    f"True: {true}\nPredicted: {pred}",
                    fontsize=9
                )

        fig.text(0.02, 0.72 if row == 0 else 0.28, title, rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0.06, 0, 1, 1])

    # Save figure
    plt.savefig(output_path, dpi=300)

    plt.close(fig)


""" EVALUATE MODEL - for methods 1-2 """

def evaluate_model(model, loader, device, num_classes):

    model.eval()

    all_true = []
    all_predictions = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Predicted classes
            predictions = outputs.argmax(dim=1)

            # Store labels and predictions
            all_true.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Convert to NumPy arrays
    y_true = np.array(all_true)
    y_pred = np.array(all_predictions)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score( y_true,  y_pred, average="macro" )

    # Confusion matrix with all 37 classes
    cm = confusion_matrix( y_true,  y_pred,  labels=np.arange(num_classes) )

    # Return all results
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }

""" CONFUSION MATRIX """

def save_confusion_matrix(cm, class_names, title, output_path):

    disp = ConfusionMatrixDisplay( confusion_matrix=cm, display_labels=class_names )

    fig, ax = plt.subplots(figsize=(18, 16))

    disp.plot(
        ax=ax,
        xticks_rotation="vertical",
        values_format="d",
        colorbar=False
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def get_top_confusions(cm, class_names, top_k=10):

    # Copy confusion matrix
    cm_errors = cm.copy()

    # Remove correct predictions (diagonal)
    np.fill_diagonal(cm_errors, 0)

    # Get indices sorted by decreasing number of errors
    sorted_indices = np.argsort(cm_errors.ravel())[::-1]

    results = []

    for idx in sorted_indices:
        count = cm_errors.ravel()[idx]

        # Stop when no more errors are available
        if count == 0:
            break

        true_idx, pred_idx = np.unravel_index(  idx, cm_errors.shape  )

        results.append({
            "True class": class_names[true_idx],
            "Predicted class": class_names[pred_idx],
            "Count": int(count)
        })

        if len(results) == top_k:
            break

    return pd.DataFrame(results)