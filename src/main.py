import matplotlib.pyplot as plt
import numpy as np

try:
    import mplcatppuccin
    import matplotlib as mpl

    mpl.style.use("mocha")
except ImportError:
    pass

from nn.activations import ReLU, Sigmoid
from nn.layers import FC, BatchNorm, Dropout
from nn.loss import BinaryCrossEntropy
from nn.model import Model
from nn.optimizers import Adam


def main() -> None:
    # Grid density
    N = 35
    # Network parameters
    EPOCHS = 50
    DROPOUT_RATE = 0.0
    LEARNING_RATE = 2.5e-3

    # Bounds
    x = np.linspace(0.0, 1.0, N)

    # Objective function that dictates the separation boundary
    y = np.ones_like(x)
    y = np.where((x >= 0.05) & (x < 0.35), np.cos(50 * np.pi * (x - 0.05)), y)
    y = np.where((x >= 0.40) & (x < 0.65), 0.25, y)
    y = np.where((x >= 0.65) & (x < 1.00), 1 - np.cos(50 * np.pi * (x - 0.65)), y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(f"Neural network visualization, {EPOCHS=}")

    # Construct the 2D grid
    w, h = np.meshgrid(x, x)

    # Display the points as two classes
    mask = h > y
    ax1.scatter(w[mask], h[mask], label="Class A")
    ax1.scatter(w[~mask], h[~mask], label="Class B")
    ax1.legend()

    # Construct training set
    X = np.dstack((w, h)).reshape(-1, 2)
    Y = mask.astype(np.float64).reshape(-1, 1)

    model = Model(
        inputs=2,
        layers=[
            # First layer
            FC(32),
            BatchNorm(),
            Dropout(DROPOUT_RATE),
            ReLU(),
            # Second layer
            FC(16),
            BatchNorm(),
            Dropout(DROPOUT_RATE),
            ReLU(),
            # Output layer
            FC(1),
            Sigmoid(),
        ],
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=BinaryCrossEntropy(),
    )

    # Train model
    model.fit(X, Y, epochs=EPOCHS, mini_batch_size=None)

    mask = mask.ravel()
    predictions = (model.predict(X) >= 0.5).ravel()
    misclassified = mask != predictions

    ax2.scatter(X[mask, 0], X[mask, 1], label="Class A")
    ax2.scatter(X[~mask, 0], X[~mask, 1], label="Class B")
    ax2.scatter(
        X[misclassified, 0],
        X[misclassified, 1],
        color="r",
        label="Misclassified",
    )

    ax1.set_title("Actual")
    ax2.set_title("Predicted")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
