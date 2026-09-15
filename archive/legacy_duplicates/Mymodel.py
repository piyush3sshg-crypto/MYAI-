import json
import time

from ai.neural import NeuralIntentClassifier
from ai.rnn import CharRNN, NeuralTopicWriter
from ai.transformer import TinyTransformer


class MyAIModel:
    def __init__(self, name="MyAI"):
        self.name = name
        self.created_at = time.time()
        self.classifier = NeuralIntentClassifier(hidden_size=16, seed=42)
        self.writer = NeuralTopicWriter(hidden_size=32, seed=42)
        self._writer_topic = "default"
        self.classifier_trained = False
        self.writer_trained = False
        self.transformer = None
        self.transformer_trained = False
        self.training_log = []

    def teach_classification(self, label, examples):
        self.classifier.add_examples(label, examples)

    def train_classifier(self, epochs=400, learning_rate=0.15, verbose=False):
        report = self.classifier.train(epochs=epochs, learning_rate=learning_rate, verbose=verbose)
        self.classifier_trained = True
        self.training_log.append({"type": "classifier", "timestamp": time.time(), **report})
        return report

    def classify(self, text, threshold=0.5):
        if not self.classifier_trained:
            raise RuntimeError(f"{self.name}: classifier has not been trained yet — call train_classifier() first")
        return self.classifier.predict(text, threshold=threshold)

    def train_writer(self, text, epochs=60, seq_length=20, learning_rate=0.15, verbose=False):
        report = self.writer.learn(
            self._writer_topic, text, epochs=epochs, seq_length=seq_length,
            learning_rate=learning_rate, verbose=verbose,
        )
        self.writer_trained = True
        self.training_log.append({"type": "writer", "timestamp": time.time(), **report})
        return report

    def generate(self, seed_text="", length=100, temperature=0.7, seed=None):
        if not self.writer_trained:
            raise RuntimeError(f"{self.name}: writer has not been trained yet — call train_writer() first")
        return self.writer.write_about(
            self._writer_topic, seed_text=seed_text, length=length, temperature=temperature, seed=seed
        )

    def train_transformer(self, text, embed_dim=16, ff_dim=32, context_length=16,
                           epochs=40, learning_rate=0.05, verbose=False):
        vocab = sorted(set(text))
        self.transformer = TinyTransformer(
            vocab, embed_dim=embed_dim, ff_dim=ff_dim, context_length=context_length, seed=42
        )
        loss_history = self.transformer.train(text, epochs=epochs, learning_rate=learning_rate, verbose=verbose)
        self.transformer_trained = True
        report = {
            "initial_loss": loss_history[0] if loss_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
            "epochs": epochs,
            "vocab_size": len(vocab),
            "text_length": len(text),
            "architecture": "self-attention transformer",
        }
        self.training_log.append({"type": "transformer", "timestamp": time.time(), **report})
        return report

    def generate_transformer(self, seed_text="", length=100, temperature=0.7, seed=None):
        if not self.transformer_trained:
            raise RuntimeError(f"{self.name}: transformer has not been trained yet — call train_transformer() first")
        return self.transformer.generate(seed_text=seed_text, length=length, temperature=temperature, seed=seed)

    def info(self):
        return {
            "name": self.name,
            "created_at": self.created_at,
            "classifier_trained": self.classifier_trained,
            "writer_trained": self.writer_trained,
            "transformer_trained": self.transformer_trained,
            "known_labels": self.classifier.intent_names if self.classifier_trained else [],
            "classifier_vocabulary_size": len(self.classifier.vocabulary) if self.classifier_trained else 0,
            "writer_vocabulary_size": (
                len(self.writer.models[self._writer_topic].vocab)
                if self.writer_trained else 0
            ),
            "transformer_vocabulary_size": len(self.transformer.vocab) if self.transformer_trained else 0,
            "total_training_runs": len(self.training_log),
            "training_log": self.training_log,
        }

    def total_parameters(self):
        count = 0
        if self.classifier_trained:
            for layer in self.classifier.network.layers:
                count += len(layer.biases)
                for row in layer.weights:
                    count += len(row)
        if self.writer_trained:
            rnn = self.writer.models[self._writer_topic]
            count += len(rnn.bh) + len(rnn.by)
            for matrix in (rnn.Wxh, rnn.Whh, rnn.Why):
                for row in matrix:
                    count += len(row)
        if self.transformer_trained:
            for name in self.transformer._param_names:
                param = getattr(self.transformer, name)
                if isinstance(param[0], list):
                    for row in param:
                        count += len(row)
                else:
                    count += len(param)
        return count

    def save(self, path):
        data = {
            "name": self.name,
            "created_at": self.created_at,
            "training_log": self.training_log,
            "classifier_trained": self.classifier_trained,
            "writer_trained": self.writer_trained,
            "transformer_trained": self.transformer_trained,
            "classifier": self.classifier.to_dict() if self.classifier_trained else None,
            "writer": (
                self.writer.models[self._writer_topic].to_dict()
                if self.writer_trained else None
            ),
            "transformer": self.transformer.to_dict() if self.transformer_trained else None,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        model = cls(name=data["name"])
        model.created_at = data["created_at"]
        model.training_log = data["training_log"]
        if data["classifier_trained"] and data["classifier"] is not None:
            model.classifier = NeuralIntentClassifier.from_dict(data["classifier"])
            model.classifier_trained = True
        if data["writer_trained"] and data["writer"] is not None:
            rnn = CharRNN.from_dict(data["writer"])
            model.writer.models[model._writer_topic] = rnn
            model.writer_trained = True
        if data.get("transformer_trained") and data.get("transformer") is not None:
            model.transformer = TinyTransformer.from_dict(data["transformer"])
            model.transformer_trained = True
        return model

    def __repr__(self):
        status = []
        if self.classifier_trained:
            status.append("classifier=trained")
        if self.writer_trained:
            status.append("writer=trained")
        if self.transformer_trained:
            status.append("transformer=trained")
        status_str = ", ".join(status) if status else "untrained"
        return f"MyAIModel(name={self.name!r}, {status_str}, params={self.total_parameters()})"
