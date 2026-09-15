import math
import random


def tanh(x):
    return math.tanh(x)


def dot(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))


def mat_vec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def outer(vec_a, vec_b):
    return [[a * b for b in vec_b] for a in vec_a]


def add_vec(vec_a, vec_b):
    return [a + b for a, b in zip(vec_a, vec_b)]


def zeros_vec(n):
    return [0.0] * n


def zeros_mat(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def softmax(values):
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps) or 1e-12
    return [e / total for e in exps]


class CharRNN:
    def __init__(self, vocab, hidden_size=48, seed=42):
        self.vocab = vocab
        self.char_to_index = {ch: i for i, ch in enumerate(vocab)}
        self.index_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size

        rng = random.Random(seed)
        scale = 0.05
        self.Wxh = [[rng.uniform(-scale, scale) for _ in range(self.vocab_size)] for _ in range(hidden_size)]
        self.Whh = [[rng.uniform(-scale, scale) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Why = [[rng.uniform(-scale, scale) for _ in range(hidden_size)] for _ in range(self.vocab_size)]
        self.bh = zeros_vec(hidden_size)
        self.by = zeros_vec(self.vocab_size)

        self.mWxh = zeros_mat(hidden_size, self.vocab_size)
        self.mWhh = zeros_mat(hidden_size, hidden_size)
        self.mWhy = zeros_mat(self.vocab_size, hidden_size)
        self.mbh = zeros_vec(hidden_size)
        self.mby = zeros_vec(self.vocab_size)

    def _one_hot(self, index):
        vec = zeros_vec(self.vocab_size)
        vec[index] = 1.0
        return vec

    def loss_and_gradients(self, input_indices, target_indices, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = list(h_prev)
        loss = 0.0

        for t, char_index in enumerate(input_indices):
            xs[t] = self._one_hot(char_index)
            raw = add_vec(mat_vec(self.Wxh, xs[t]), mat_vec(self.Whh, hs[t - 1]))
            raw = add_vec(raw, self.bh)
            hs[t] = [tanh(v) for v in raw]
            ys[t] = add_vec(mat_vec(self.Why, hs[t]), self.by)
            ps[t] = softmax(ys[t])
            loss += -math.log(max(ps[t][target_indices[t]], 1e-12))

        dWxh = zeros_mat(self.hidden_size, self.vocab_size)
        dWhh = zeros_mat(self.hidden_size, self.hidden_size)
        dWhy = zeros_mat(self.vocab_size, self.hidden_size)
        dbh = zeros_vec(self.hidden_size)
        dby = zeros_vec(self.vocab_size)
        dh_next = zeros_vec(self.hidden_size)

        for t in reversed(range(len(input_indices))):
            dy = list(ps[t])
            dy[target_indices[t]] -= 1.0
            dWhy = [[dWhy[i][j] + dy[i] * hs[t][j] for j in range(self.hidden_size)] for i in range(self.vocab_size)]
            dby = add_vec(dby, dy)

            dh = add_vec(mat_vec(self._transpose(self.Why), dy), dh_next)
            dh_raw = [(1 - hs[t][i] ** 2) * dh[i] for i in range(self.hidden_size)]
            dbh = add_vec(dbh, dh_raw)
            dWxh = [[dWxh[i][j] + dh_raw[i] * xs[t][j] for j in range(self.vocab_size)] for i in range(self.hidden_size)]
            dWhh = [[dWhh[i][j] + dh_raw[i] * hs[t - 1][j] for j in range(self.hidden_size)] for i in range(self.hidden_size)]
            dh_next = mat_vec(self._transpose(self.Whh), dh_raw)

        for grad_mat in (dWxh, dWhh, dWhy):
            for row in grad_mat:
                for i in range(len(row)):
                    row[i] = max(-5.0, min(5.0, row[i]))
        for grad_vec in (dbh, dby):
            for i in range(len(grad_vec)):
                grad_vec[i] = max(-5.0, min(5.0, grad_vec[i]))

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(input_indices) - 1]

    def _transpose(self, matrix):
        if not matrix:
            return []
        return [[matrix[r][c] for r in range(len(matrix))] for c in range(len(matrix[0]))]

    def _adagrad_update(self, param, grad, memory, learning_rate):
        if isinstance(param[0], list):
            for i in range(len(param)):
                for j in range(len(param[i])):
                    memory[i][j] += grad[i][j] ** 2
                    param[i][j] -= learning_rate * grad[i][j] / math.sqrt(memory[i][j] + 1e-8)
        else:
            for i in range(len(param)):
                memory[i] += grad[i] ** 2
                param[i] -= learning_rate * grad[i] / math.sqrt(memory[i] + 1e-8)

    def train(self, text, seq_length=20, epochs=50, learning_rate=0.1, verbose=False):
        indices = [self.char_to_index[ch] for ch in text if ch in self.char_to_index]
        if len(indices) <= seq_length:
            raise ValueError("training text too short for the given seq_length")
        loss_history = []
        for epoch in range(epochs):
            h_prev = zeros_vec(self.hidden_size)
            position = 0
            epoch_loss = 0.0
            steps = 0
            while position + seq_length + 1 <= len(indices):
                input_chunk = indices[position:position + seq_length]
                target_chunk = indices[position + 1:position + seq_length + 1]
                loss, dWxh, dWhh, dWhy, dbh, dby, h_prev = self.loss_and_gradients(
                    input_chunk, target_chunk, h_prev
                )
                self._adagrad_update(self.Wxh, dWxh, self.mWxh, learning_rate)
                self._adagrad_update(self.Whh, dWhh, self.mWhh, learning_rate)
                self._adagrad_update(self.Why, dWhy, self.mWhy, learning_rate)
                self._adagrad_update(self.bh, dbh, self.mbh, learning_rate)
                self._adagrad_update(self.by, dby, self.mby, learning_rate)
                epoch_loss += loss
                steps += 1
                position += seq_length
            average_loss = epoch_loss / max(1, steps)
            loss_history.append(average_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  loss {average_loss:.4f}")
        return loss_history

    def sample(self, seed_text="", length=100, temperature=0.7, seed=None):
        rng = random.Random(seed)
        h = zeros_vec(self.hidden_size)
        output_chars = []
        if seed_text:
            for ch in seed_text:
                if ch not in self.char_to_index:
                    continue
                x = self._one_hot(self.char_to_index[ch])
                raw = add_vec(add_vec(mat_vec(self.Wxh, x), mat_vec(self.Whh, h)), self.bh)
                h = [tanh(v) for v in raw]
                output_chars.append(ch)
            last_char = seed_text[-1] if seed_text[-1] in self.char_to_index else rng.choice(self.vocab)
        else:
            last_char = rng.choice(self.vocab)
            output_chars.append(last_char)

        current_index = self.char_to_index[last_char]
        for _ in range(length):
            x = self._one_hot(current_index)
            raw = add_vec(add_vec(mat_vec(self.Wxh, x), mat_vec(self.Whh, h)), self.bh)
            h = [tanh(v) for v in raw]
            y = add_vec(mat_vec(self.Why, h), self.by)
            scaled = [v / max(temperature, 1e-6) for v in y]
            probabilities = softmax(scaled)
            r = rng.random()
            cumulative = 0.0
            chosen_index = len(probabilities) - 1
            for i, p in enumerate(probabilities):
                cumulative += p
                if r <= cumulative:
                    chosen_index = i
                    break
            output_chars.append(self.index_to_char[chosen_index])
            current_index = chosen_index
        return "".join(output_chars)


class NeuralTopicWriter:
    def __init__(self, hidden_size=48, seed=42):
        self.hidden_size = hidden_size
        self.seed = seed
        self.models = {}
        self.training_reports = {}

    def learn(self, topic_name, text, seq_length=20, epochs=60, learning_rate=0.1, verbose=False):
        vocab = sorted(set(text))
        if len(vocab) < 2:
            raise ValueError("training text needs at least 2 distinct characters")
        model = CharRNN(vocab, hidden_size=self.hidden_size, seed=self.seed)
        loss_history = model.train(
            text, seq_length=seq_length, epochs=epochs, learning_rate=learning_rate, verbose=verbose
        )
        self.models[topic_name] = model
        self.training_reports[topic_name] = {
            "initial_loss": loss_history[0] if loss_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
            "epochs": epochs,
            "vocab_size": len(vocab),
            "text_length": len(text),
        }
        return self.training_reports[topic_name]

    def known_topics(self):
        return list(self.models.keys())

    def write_about(self, topic_name, seed_text="", length=120, temperature=0.7, seed=None):
        if topic_name not in self.models:
            return None
        return self.models[topic_name].sample(
            seed_text=seed_text, length=length, temperature=temperature, seed=seed
        )
