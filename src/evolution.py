import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

try:
    import mplcatppuccin
    import matplotlib as mpl

    mpl.style.use("mocha")
except ImportError:
    pass

from nn.activations import ReLU, Sigmoid
from nn.layers import FC, BatchNorm
from nn.loss import BinaryCrossEntropy
from nn.model import Model
from nn.optimizers import Adam


def main() -> None:
    # Grid density
    N = 75
    # Hyperparameters
    EPOCHS = 7_500
    LEARNING_RATE = 5e-3

    # Bounds
    x = np.linspace(0.0, 1.0, N)

    # Objective function that dictates the separation boundary
    y = np.ones_like(x)
    y = np.where((x < 0.35), np.tanh(7.5 * x), y)
    y = np.where((x >= 0.45) & (x < 0.55), 0, y)
    y = np.where((x >= 0.65) & (x <= 1.00), 1 - np.tanh(5 * (x - 0.65)), y)

    # Construct the 2D grid
    w, h = np.meshgrid(x, x)

    # Construct training set
    mask = (h >= y).ravel()
    X = np.dstack((w, h)).reshape(-1, 2)
    Y = mask.astype(np.float64).reshape(-1, 1)

    model = Model(
        inputs=2,
        layers=[
            # First layer
            FC(32),
            BatchNorm(),
            ReLU(),
            # Second layer
            FC(8),
            BatchNorm(),
            ReLU(),
            # Third layer
            FC(8),
            BatchNorm(),
            ReLU(),
            # Output layer
            FC(1),
            Sigmoid(),
        ],
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=BinaryCrossEntropy(),
    )

    fig, ax = plt.subplots()

    im = ax.imshow(h, origin="lower")
    title = ax.text(
        0.875,
        0.965,
        "Placeholder",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        bbox={"facecolor": "black", "alpha": 0.5},
    )

    def update(epoch):
        model.fit(X, Y, epochs=1, mini_batch_size=None, verbose=False)
        predictions = (model.predict(X) >= 0.5).ravel()
        misclassified = mask != predictions

        title.set_text(f"{epoch=}")

        colors = np.where(mask, 1.0, 0.5)
        colors[misclassified] = 0

        im.set_data(colors.reshape(N, N))

        return im, title

    fig.tight_layout()
    FuncAnimation(fig, update, frames=range(EPOCHS), blit=True, interval=10)

    plt.show()


if __name__ == "__main__":
    main()
