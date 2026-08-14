import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_emotion_probabilities(predictions, class_names):
    """
    Plots a horizontal bar chart of predicted probabilities for each emotion class.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("viridis", len(class_names))
    y_pos = np.arange(len(class_names))

    ax.barh(y_pos, predictions, align="center", color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title("Facial Emotion Detection Confidence")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    Plots training and validation accuracy and loss curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig
