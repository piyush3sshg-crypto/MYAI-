import random

from .neural import FeedForwardNetwork

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_image_as_grid(path, size=(8, 8)):
    if not PIL_AVAILABLE:
        raise RuntimeError(
            "PIL is not installed, so real photo files can't be read. "
            "In Termux run: pkg install python-pillow  (then restart python). "
            "Pixel-grid images (plain lists of numbers) work without this."
        )
    img = Image.open(path).convert("L")
    img = img.resize(size)
    pixels = list(img.getdata())
    return [p / 255.0 for p in pixels]


def grid_from_rows(rows):
    flat = []
    for row in rows:
        flat.extend(row)
    return [float(v) for v in flat]


def _flatten_input(grid_or_flat, expected_len):
    if grid_or_flat and isinstance(grid_or_flat[0], (list, tuple)):
        flat = grid_from_rows(grid_or_flat)
    else:
        flat = [float(v) for v in grid_or_flat]
    if len(flat) != expected_len:
        raise ValueError(f"expected {expected_len} pixel values, got {len(flat)}")
    return flat


class TinyVisionClassifier:
    def __init__(self, grid_size=(8, 8), hidden_size=16, seed=42):
        self.grid_size = tuple(grid_size)
        self.input_dim = self.grid_size[0] * self.grid_size[1]
        self.hidden_size = hidden_size
        self.seed = seed
        self.labels = []
        self.raw_examples = []
        self.network = None
        self.trained = False
        self.last_training_report = None

    def add_example(self, label, grid_or_flat):
        flat = _flatten_input(grid_or_flat, self.input_dim)
        self.raw_examples.append((label, flat))

    def add_example_from_file(self, label, path):
        flat = load_image_as_grid(path, size=self.grid_size)
        self.raw_examples.append((label, flat))

    def train(self, epochs=300, learning_rate=0.15, verbose=False):
        if not self.raw_examples:
            raise ValueError("no training examples registered")
        self.labels = sorted({label for label, _ in self.raw_examples})
        label_to_index = {label: i for i, label in enumerate(self.labels)}
        self.network = FeedForwardNetwork(
            [self.input_dim, self.hidden_size, len(self.labels)], seed=self.seed
        )
        training_data = [(vec, label_to_index[label]) for label, vec in self.raw_examples]
        rng = random.Random(self.seed)
        loss_history = []
        for epoch in range(epochs):
            rng.shuffle(training_data)
            epoch_loss = 0.0
            for x, target in training_data:
                epoch_loss += self.network.train_step(x, target, learning_rate)
            average_loss = epoch_loss / len(training_data)
            loss_history.append(average_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  loss {average_loss:.5f}")
        correct = sum(
            1 for x, target in training_data if self.network.predict_index(x)[0] == target
        )
        accuracy = correct / len(training_data)
        self.trained = True
        self.last_training_report = {
            "final_loss": loss_history[-1] if loss_history else None,
            "training_accuracy": accuracy,
            "examples": len(training_data),
            "labels": self.labels,
            "grid_size": self.grid_size,
        }
        return self.last_training_report

    def classify(self, grid_or_flat_or_path, from_file=False):
        if not self.trained:
            raise RuntimeError("vision classifier has not been trained yet")
        if from_file:
            flat = load_image_as_grid(grid_or_flat_or_path, size=self.grid_size)
        else:
            flat = _flatten_input(grid_or_flat_or_path, self.input_dim)
        index, confidence, _ = self.network.predict_index(flat)
        return self.labels[index], confidence

    def to_dict(self):
        layers_data = []
        for layer in self.network.layers:
            layers_data.append({
                "weights": layer.weights,
                "biases": layer.biases,
                "activation": layer.activation_name,
            })
        return {
            "grid_size": list(self.grid_size),
            "hidden_size": self.hidden_size,
            "seed": self.seed,
            "labels": self.labels,
            "layers": layers_data,
            "last_training_report": self.last_training_report,
        }

    @classmethod
    def from_dict(cls, data):
        clf = cls(grid_size=tuple(data["grid_size"]), hidden_size=data["hidden_size"], seed=data["seed"])
        clf.labels = data["labels"]
        clf.last_training_report = data.get("last_training_report")
        layer_sizes = [clf.input_dim, clf.hidden_size, len(clf.labels)]
        clf.network = FeedForwardNetwork(layer_sizes, seed=clf.seed)
        for layer_obj, layer_data in zip(clf.network.layers, data["layers"]):
            layer_obj.weights = layer_data["weights"]
            layer_obj.biases = layer_data["biases"]
        clf.trained = True
        return clf

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
