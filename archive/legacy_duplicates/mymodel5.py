import json
import time

from ai.neural import NeuralIntentClassifier
from ai.rnn import CharRNN, NeuralTopicWriter
from ai.transformer import TinyTransformer
from ai.vision import TinyVisionClassifier, PIL_AVAILABLE


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
        self.vision = TinyVisionClassifier(grid_size=(8, 8), hidden_size=16, seed=42)
        self.vision_trained = False
        self.training_log = []
        self._writer_corpus = ""
        self._transformer_corpus = ""
        self.long_term_memory = {}
        self.interaction_log = []
        self.correction_count = 0

    def teach_classification(self, label, examples):
        self.classifier.add_examples(label, examples)

    def train_classifier(self, epochs=400, learning_rate=0.15, verbose=False, validation_split=0.2):
        report = self.classifier.train(
            epochs=epochs, learning_rate=learning_rate, verbose=verbose, validation_split=validation_split
        )
        self.classifier_trained = True
        self.training_log.append({"type": "classifier", "timestamp": time.time(), **report})
        return report

    def classify(self, text, threshold=0.5):
        if not self.classifier_trained:
            raise RuntimeError(f"{self.name}: classifier has not been trained yet — call train_classifier() first")
        result = self.classifier.predict(text, threshold=threshold)
        self._log_interaction("classify", text, {"label": result.intent_name, "confidence": result.score})
        return result

    def correct(self, text, correct_label, retrain_epochs=150, retrain_learning_rate=0.15):
        self.classifier.add_example(correct_label, text)
        self.correction_count += 1
        report = self.train_classifier(epochs=retrain_epochs, learning_rate=retrain_learning_rate)
        self._log_interaction("correction", text, {"correct_label": correct_label})
        return report

    def remember(self, key, value):
        self.long_term_memory[key] = {"value": value, "timestamp": time.time()}
        self._log_interaction("remember", key, value)

    def recall(self, key, default=None):
        entry = self.long_term_memory.get(key)
        return entry["value"] if entry is not None else default

    def forget(self, key):
        if key in self.long_term_memory:
            del self.long_term_memory[key]
            return True
        return False

    def known_facts(self):
        return list(self.long_term_memory.keys())

    def _log_interaction(self, interaction_type, input_data, output_data):
        self.interaction_log.append({
            "type": interaction_type,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data,
        })
        if len(self.interaction_log) > 1000:
            self.interaction_log = self.interaction_log[-1000:]

    def learning_history(self, n=20):
        return self.interaction_log[-n:]

    def growth_summary(self):
        return {
            "name": self.name,
            "age_seconds": time.time() - self.created_at,
            "total_interactions": len(self.interaction_log),
            "corrections_made": self.correction_count,
            "long_term_facts_remembered": len(self.long_term_memory),
            "training_runs": len(self.training_log),
            "current_parameters": self.total_parameters(),
        }

    def self_evaluate(self):
        report = {"name": self.name, "checks": []}

        if self.classifier_trained:
            last_report = self.classifier.last_training_report or {}
            train_acc = last_report.get("training_accuracy")
            val_acc = last_report.get("validation_accuracy")
            check = {
                "component": "classifier",
                "training_accuracy": train_acc,
                "validation_accuracy": val_acc,
            }
            if val_acc is not None and train_acc is not None:
                gap = train_acc - val_acc
                check["train_validation_gap"] = gap
                if gap > 0.25:
                    check["diagnosis"] = (
                        "Likely OVERFITTING — the classifier memorizes training text well "
                        "but performs much worse on unseen text. More diverse training "
                        "examples per label would help."
                    )
                elif gap > 0.1:
                    check["diagnosis"] = "Mild overfitting — generalizes reasonably but could improve with more data."
                else:
                    check["diagnosis"] = "Generalizes well — training and validation accuracy are close."
            else:
                check["diagnosis"] = (
                    "No validation split was used, so generalization cannot be assessed. "
                    "Retrain with validation_split > 0 for an honest estimate."
                )

            calibration = self.classifier.evaluate_calibration(num_buckets=5)
            check["calibration_error"] = calibration["calibration_error"]
            if calibration["calibration_error"] is not None:
                if calibration["calibration_error"] > 0.15:
                    check["calibration_diagnosis"] = (
                        "Confidence scores are POORLY CALIBRATED — do not fully trust the "
                        "confidence numbers, especially for borderline predictions."
                    )
                else:
                    check["calibration_diagnosis"] = "Confidence scores are reasonably calibrated."
            report["checks"].append(check)

        if self.vision_trained:
            v_report = self.vision.last_training_report or {}
            report["checks"].append({
                "component": "vision",
                "training_accuracy": v_report.get("training_accuracy"),
                "diagnosis": (
                    "No validation split implemented for vision yet — "
                    "training accuracy alone should not be fully trusted."
                ),
            })

        if self.writer_trained:
            w_report = self.writer.training_reports.get(self._writer_topic, {})
            report["checks"].append({
                "component": "writer",
                "final_loss": w_report.get("final_loss"),
                "diagnosis": (
                    "Low loss on training text does not guarantee good generation quality — "
                    "always sample and read the output yourself."
                ),
            })

        if self.transformer_trained:
            report["checks"].append({
                "component": "transformer",
                "diagnosis": "Same caveat as writer — inspect generated samples directly, loss alone is not sufficient.",
            })

        report["overall_note"] = (
            "This is a small, self-trained model. Self-evaluation checks help catch "
            "overfitting and miscalibration, but cannot substitute for testing on real, "
            "diverse, unseen inputs before trusting the model."
        )
        return report

    def train_writer(self, text, epochs=60, seq_length=20, learning_rate=0.15, verbose=False):
        self._writer_corpus = (self._writer_corpus + " " + text).strip() if self._writer_corpus else text
        report = self.writer.learn(
            self._writer_topic, self._writer_corpus, epochs=epochs, seq_length=seq_length,
            learning_rate=learning_rate, verbose=verbose,
        )
        self.writer_trained = True
        report["cumulative_corpus_length"] = len(self._writer_corpus)
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
        self._transformer_corpus = (
            (self._transformer_corpus + " " + text).strip() if self._transformer_corpus else text
        )
        vocab = sorted(set(self._transformer_corpus))
        self.transformer = TinyTransformer(
            vocab, embed_dim=embed_dim, ff_dim=ff_dim, context_length=context_length, seed=42
        )
        loss_history = self.transformer.train(
            self._transformer_corpus, epochs=epochs, learning_rate=learning_rate, verbose=verbose
        )
        self.transformer_trained = True
        report = {
            "initial_loss": loss_history[0] if loss_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
            "epochs": epochs,
            "vocab_size": len(vocab),
            "text_length": len(self._transformer_corpus),
            "cumulative_corpus_length": len(self._transformer_corpus),
            "architecture": "self-attention transformer",
        }
        self.training_log.append({"type": "transformer", "timestamp": time.time(), **report})
        return report

    def generate_transformer(self, seed_text="", length=100, temperature=0.7, seed=None):
        if not self.transformer_trained:
            raise RuntimeError(f"{self.name}: transformer has not been trained yet — call train_transformer() first")
        return self.transformer.generate(seed_text=seed_text, length=length, temperature=temperature, seed=seed)

    def teach_vision(self, label, grid_or_flat):
        self.vision.add_example(label, grid_or_flat)

    def teach_vision_from_file(self, label, path):
        self.vision.add_example_from_file(label, path)

    def train_vision(self, epochs=300, learning_rate=0.15, verbose=False):
        report = self.vision.train(epochs=epochs, learning_rate=learning_rate, verbose=verbose)
        self.vision_trained = True
        self.training_log.append({"type": "vision", "timestamp": time.time(), **report})
        return report

    def classify_image(self, grid_or_flat_or_path, from_file=False):
        if not self.vision_trained:
            raise RuntimeError(f"{self.name}: vision has not been trained yet — call train_vision() first")
        return self.vision.classify(grid_or_flat_or_path, from_file=from_file)

    def understand_multimodal(self, text=None, image=None, image_is_path=False):
        result = {"name": self.name}
        if text is not None:
            if not self.classifier_trained:
                result["text_error"] = "classifier not trained — call train_classifier() first"
            else:
                match = self.classify(text)
                result["text"] = {"label": match.intent_name, "confidence": match.score}
        if image is not None:
            if not self.vision_trained:
                result["image_error"] = "vision not trained — call train_vision() first"
            else:
                label, confidence = self.classify_image(image, from_file=image_is_path)
                result["image"] = {"label": label, "confidence": confidence}
        if "text" in result and "image" in result:
            result["fusion_note"] = (
                f"Text signal says {result['text']['label']!r}, "
                f"image signal says {result['image']['label']!r}."
            )
        return result

    def info(self):
        return {
            "name": self.name,
            "created_at": self.created_at,
            "classifier_trained": self.classifier_trained,
            "writer_trained": self.writer_trained,
            "transformer_trained": self.transformer_trained,
            "vision_trained": self.vision_trained,
            "pil_available_for_real_photos": PIL_AVAILABLE,
            "known_labels": self.classifier.intent_names if self.classifier_trained else [],
            "classifier_vocabulary_size": len(self.classifier.vocabulary) if self.classifier_trained else 0,
            "writer_vocabulary_size": (
                len(self.writer.models[self._writer_topic].vocab)
                if self.writer_trained else 0
            ),
            "transformer_vocabulary_size": len(self.transformer.vocab) if self.transformer_trained else 0,
            "vision_labels": self.vision.labels if self.vision_trained else [],
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
        if self.vision_trained:
            for layer in self.vision.network.layers:
                count += len(layer.biases)
                for row in layer.weights:
                    count += len(row)
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
            "vision_trained": self.vision_trained,
            "vision": self.vision.to_dict() if self.vision_trained else None,
            "long_term_memory": self.long_term_memory,
            "interaction_log": self.interaction_log,
            "correction_count": self.correction_count,
            "writer_corpus": self._writer_corpus,
            "transformer_corpus": self._transformer_corpus,
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
        if data.get("vision_trained") and data.get("vision") is not None:
            model.vision = TinyVisionClassifier.from_dict(data["vision"])
            model.vision_trained = True
        model.long_term_memory = data.get("long_term_memory", {})
        model.interaction_log = data.get("interaction_log", [])
        model.correction_count = data.get("correction_count", 0)
        model._writer_corpus = data.get("writer_corpus", "")
        model._transformer_corpus = data.get("transformer_corpus", "")
        return model

    def __repr__(self):
        status = []
        if self.classifier_trained:
            status.append("classifier=trained")
        if self.writer_trained:
            status.append("writer=trained")
        if self.transformer_trained:
            status.append("transformer=trained")
        if self.vision_trained:
            status.append("vision=trained")
        status_str = ", ".join(status) if status else "untrained"
        return f"MyAIModel(name={self.name!r}, {status_str}, params={self.total_parameters()})"
