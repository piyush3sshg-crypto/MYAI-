import math
import random

from core.tokenizee import Tokenizer


def _xavier_limit(fan_in, fan_out):
    return math.sqrt(6.0 / (fan_in + fan_out))


def tanh(x):
    return math.tanh(x)


def tanh_derivative(activated_value):
    return 1.0 - activated_value * activated_value


def relu(x):
    return x if x > 0 else 0.0


def relu_derivative(activated_value):
    return 1.0 if activated_value > 0 else 0.0


def softmax(values):
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps) or 1e-12
    return [e / total for e in exps]


def cross_entropy_loss(predicted, target_index):
    p = max(predicted[target_index], 1e-12)
    return -math.log(p)


ACTIVATIONS = {
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
}


class DenseLayer:
    def __init__(self, input_dim, output_dim, activation="tanh", seed=None):
        rng = random.Random(seed)
        limit = _xavier_limit(input_dim, output_dim)
        self.weights = [
            [rng.uniform(-limit, limit) for _ in range(input_dim)]
            for _ in range(output_dim)
        ]
        self.biases = [0.0 for _ in range(output_dim)]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.activation_fn, self.activation_derivative_fn = ACTIVATIONS.get(
            activation, (None, None)
        )
        self.last_input = None
        self.last_z = None
        self.last_a = None

    def forward(self, x):
        self.last_input = x
        z = []
        for j in range(self.output_dim):
            total = self.biases[j]
            row = self.weights[j]
            for i in range(self.input_dim):
                total += row[i] * x[i]
            z.append(total)
        self.last_z = z
        if self.activation_fn is not None:
            a = [self.activation_fn(v) for v in z]
        else:
            a = z
        self.last_a = a
        return a

    def backward(self, dA, learning_rate):
        if self.activation_derivative_fn is not None:
            dz = [dA[j] * self.activation_derivative_fn(self.last_a[j]) for j in range(self.output_dim)]
        else:
            dz = list(dA)
        dx = [0.0] * self.input_dim
        for j in range(self.output_dim):
            grad_j = dz[j]
            row = self.weights[j]
            for i in range(self.input_dim):
                dx[i] += row[i] * grad_j
                row[i] -= learning_rate * grad_j * self.last_input[i]
            self.biases[j] -= learning_rate * grad_j
        return dx

    def backward_from_output_gradient(self, dz, learning_rate):
        dx = [0.0] * self.input_dim
        for j in range(self.output_dim):
            grad_j = dz[j]
            row = self.weights[j]
            for i in range(self.input_dim):
                dx[i] += row[i] * grad_j
                row[i] -= learning_rate * grad_j * self.last_input[i]
            self.biases[j] -= learning_rate * grad_j
        return dx


class FeedForwardNetwork:
    def __init__(self, layer_sizes, hidden_activation="tanh", seed=42):
        self.layers = []
        rng = random.Random(seed)
        for idx in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[idx]
            output_dim = layer_sizes[idx + 1]
            is_output_layer = idx == len(layer_sizes) - 2
            activation = None if is_output_layer else hidden_activation
            layer = DenseLayer(
                input_dim, output_dim, activation=activation, seed=rng.randint(0, 999999)
            )
            self.layers.append(layer)

    def forward(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward(activation)
        return softmax(activation)

    def train_step(self, x, target_index, learning_rate):
        raw_output = x
        for layer in self.layers:
            raw_output = layer.forward(raw_output)
        probabilities = softmax(raw_output)
        loss = cross_entropy_loss(probabilities, target_index)
        output_gradient = list(probabilities)
        output_gradient[target_index] -= 1.0
        grad = self.layers[-1].backward_from_output_gradient(output_gradient, learning_rate)
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad, learning_rate)
        return loss

    def predict_index(self, x):
        probabilities = self.forward(x)
        best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        return best_index, probabilities[best_index], probabilities


class NeuralIntentMatch:
    def __init__(self, intent_name, score):
        self.intent_name = intent_name
        self.score = score
        self.entities = {}

    def __repr__(self):
        return f"NeuralIntentMatch({self.intent_name!r}, score={self.score:.4f})"


class NeuralIntentClassifier:
    def __init__(self, hidden_size=16, hidden_activation="tanh", seed=42):
        self.hidden_size = hidden_size
        self.hidden_activation = hidden_activation
        self.seed = seed
        self.tokenizer = Tokenizer(strip_stopwords=True, apply_stemming=True)
        self.raw_examples = []
        self.vocabulary = {}
        self.intent_names = []
        self.intent_to_index = {}
        self.network = None
        self.trained = False
        self.last_training_report = None

    def add_example(self, intent_name, text):
        self.raw_examples.append((intent_name, text))

    def add_examples(self, intent_name, texts):
        for text in texts:
            self.add_example(intent_name, text)

    def _build_vocabulary(self):
        vocab_set = set()
        for _, text in self.raw_examples:
            tokens = self.tokenizer.process(text)
            vocab_set.update(tokens)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab_set))}

    def _vectorize(self, text):
        tokens = self.tokenizer.process(text)
        vector = [0.0] * len(self.vocabulary)
        if not tokens:
            return vector
        for token in tokens:
            if token in self.vocabulary:
                vector[self.vocabulary[token]] += 1.0
        total = float(len(tokens))
        return [v / total for v in vector]

    def train(self, epochs=300, learning_rate=0.15, verbose=False, validation_split=0.0):
        if not self.raw_examples:
            raise ValueError("no training examples registered")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be between 0.0 and 1.0")

        self._build_vocabulary()
        self.intent_names = sorted({intent for intent, _ in self.raw_examples})
        self.intent_to_index = {
            name: idx for idx, name in enumerate(self.intent_names)
        }

        input_dim = len(self.vocabulary)
        output_dim = len(self.intent_names)

        self.network = FeedForwardNetwork(
            [input_dim, self.hidden_size, output_dim],
            hidden_activation=self.hidden_activation,
            seed=self.seed,
        )

        all_data = [
            (self._vectorize(text), self.intent_to_index[intent], text)
            for intent, text in self.raw_examples
        ]

        rng = random.Random(self.seed)
        rng.shuffle(all_data)

        val_count = int(len(all_data) * validation_split)
        validation_data = all_data[:val_count]
        training_data = [
            (x, target_index)
            for x, target_index, _ in (
                all_data[val_count:] if val_count > 0 else all_data
            )
        ]

        if not training_data:
            raise ValueError("validation_split leaves no training examples")

        loss_history = []

        for epoch in range(epochs):
            rng.shuffle(training_data)
            epoch_loss = 0.0

            for x, target_index in training_data:
                epoch_loss += self.network.train_step(
                    x, target_index, learning_rate
                )

            average_loss = epoch_loss / len(training_data)
            loss_history.append(average_loss)

            if verbose and (
                epoch % max(1, epochs // 10) == 0
                or epoch == epochs - 1
            ):
                print(f"epoch {epoch:4d}  loss {average_loss:.5f}")

        def accuracy(dataset):
            if not dataset:
                return None

            correct = 0
            for x, target_index in dataset:
                predicted_index, _, _ = self.network.predict_index(x)
                if predicted_index == target_index:
                    correct += 1

            return correct / len(dataset)

        training_accuracy = accuracy(training_data)
        validation_accuracy = accuracy(
            [(x, target_index) for x, target_index, _ in validation_data]
        )

        self.last_training_report = {
            "final_loss": loss_history[-1] if loss_history else None,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "validation_examples": len(validation_data),
            "examples": len(training_data),
            "vocabulary_size": input_dim,
            "intents": self.intent_names,
        }

        self.trained = True
        return self.last_training_report

    def predict(self, text, threshold=0.0):
        if not self.trained:
            raise RuntimeError("neural intent classifier has not been trained yet")
        vector = self._vectorize(text)
        if sum(vector) == 0.0:
            return NeuralIntentMatch("unknown", 0.0)
        predicted_index, confidence, probabilities = self.network.predict_index(vector)
        intent_name = self.intent_names[predicted_index]
        if confidence < threshold:
            return NeuralIntentMatch("unknown", confidence)
        return NeuralIntentMatch(intent_name, confidence)

    def predict_distribution(self, text):
        if not self.trained:
            raise RuntimeError("neural intent classifier has not been trained yet")
        vector = self._vectorize(text)
        _, _, probabilities = self.network.predict_index(vector)
        return dict(zip(self.intent_names, probabilities))

    def to_dict(self):
        layers_data = []
        for layer in self.network.layers:
            layers_data.append({
                "weights": layer.weights,
                "biases": layer.biases,
                "activation": layer.activation_name,
                "input_dim": layer.input_dim,
                "output_dim": layer.output_dim,
            })
        return {
            "hidden_size": self.hidden_size,
            "hidden_activation": self.hidden_activation,
            "seed": self.seed,
            "vocabulary": self.vocabulary,
            "intent_names": self.intent_names,
            "intent_to_index": self.intent_to_index,
            "layers": layers_data,
            "last_training_report": self.last_training_report,
        }

    @classmethod
    def from_dict(cls, data):
        classifier = cls(hidden_size=data["hidden_size"], hidden_activation=data["hidden_activation"], seed=data["seed"])
        classifier.vocabulary = data["vocabulary"]
        classifier.intent_names = data["intent_names"]
        classifier.intent_to_index = data["intent_to_index"]
        classifier.last_training_report = data.get("last_training_report")
        layer_sizes = [data["layers"][0]["input_dim"]] + [layer["output_dim"] for layer in data["layers"]]
        classifier.network = FeedForwardNetwork(layer_sizes, hidden_activation=data["hidden_activation"], seed=data["seed"])
        for layer_obj, layer_data in zip(classifier.network.layers, data["layers"]):
            layer_obj.weights = layer_data["weights"]
            layer_obj.biases = layer_data["biases"]
        classifier.trained = True
        return classifier

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
