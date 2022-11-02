import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_percentage_error


def log_mean_average_error(y_true, y_pred):
    return np.mean(np.abs(np.log(y_true) - np.log(y_pred)))


def calculate_metrics(y_true, y_pred):
    return {
        "min": np.min(y_pred),
        "max": np.max(y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "log_mae": log_mean_average_error(y_true, y_pred),
    }


def print_metrics(y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)
    print("Min: ", metrics["min"])
    print("Max: ", metrics["max"])
    print("R2: ", metrics["r2"])
    print("MAPE: ", metrics["mape"])
    print("Log MAE: ", metrics["log_mae"])


def plot_scatterplot(y_true: np.ndarray, y_pred: np.ndarray, logy=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=y_true.flatten(), y=y_pred.flatten(), ax=ax)

    if logy:
        ax.set_yscale('log')

    return fig


def plot_distributions(train_prices: np.ndarray, test_prices: np.ndarray,
                       predicted_prices: np.ndarray):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    sns.histplot(train_prices.flatten(), ax=ax[0], bins=50)
    ax[0].set_title("Train Data")
    sns.histplot(test_prices.flatten(), ax=ax[1], bins=50)
    ax[1].set_title("Test Data")
    sns.histplot(predicted_prices.flatten(), ax=ax[2], bins=50)
    ax[2].set_title("Predicted")

    return fig


def plot_all(test_prices: np.ndarray,
             predicted_prices: np.ndarray,
             max_ticks=None):
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))

    sns.scatterplot(x=test_prices.flatten(),
                    y=predicted_prices.flatten(),
                    ax=ax[0, 0])
    ax[0, 0].set_title("True vs Predicted")
    sns.scatterplot(x=test_prices.flatten(),
                    y=predicted_prices.flatten(),
                    ax=ax[0, 1])
    ax[0, 1].set_title("True vs Predicted (log)")
    ax[0, 1].set_yscale('log')

    sns.histplot(test_prices.flatten(), ax=ax[1, 0], bins=50)
    ax[1, 0].set_title("Test Data")
    sns.histplot(predicted_prices.flatten(), ax=ax[1, 1], bins=50)
    ax[1, 1].set_title("Predicted")
    ax[1, 0].sharey(ax[1, 1])

    if max_ticks is not None:
        for c_ax in ax.flat:
            c_ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))

    fig.tight_layout()
    return fig


# plot tensorflow loss
def plot_loss(history):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    return fig
