import math
import random


def zeros_mat(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def zeros_vec(n):
    return [0.0] * n


def random_mat(rows, cols, scale, rng):
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def matmul(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError(f"shape mismatch {cols_a} vs {rows_b}")
    result = zeros_mat(rows_a, cols_b)
    for i in range(rows_a):
        row_a = a[i]
        for k in range(cols_a):
            v = row_a[k]
            if v == 0.0:
                continue
            row_b = b[k]
            result_row = result[i]
            for j in range(cols_b):
                result_row[j] += v * row_b[j]
    return result


def transpose(a):
    if not a:
        return []
    return [[a[r][c] for r in range(len(a))] for c in range(len(a[0]))]


def add_mat(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def scale_mat(a, s):
    return [[v * s for v in row] for row in a]


def add_row_broadcast(a, vec):
    return [[a[i][j] + vec[j] for j in range(len(vec))] for i in range(len(a))]


def relu_mat(a):
    return [[v if v > 0 else 0.0 for v in row] for row in a]


def relu_grad_mat(a):
    return [[1.0 if v > 0 else 0.0 for v in row] for row in a]


def elementwise_mul(a, b):
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def softmax_rows(matrix, mask=None):
    result = []
    for i, row in enumerate(matrix):
        if mask is not None:
            row = [v if mask[i][j] else -1e9 for j, v in enumerate(row)]
        max_val = max(row)
        exps = [math.exp(v - max_val) for v in row]
        total = sum(exps) or 1e-12
        result.append([e / total for e in exps])
    return result


def causal_mask(seq_len):
    return [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]


class TinyTransformer:
    def __init__(self, vocab, embed_dim=16, ff_dim=32, context_length=16, seed=42):
        self.vocab = vocab
        self.char_to_index = {ch: i for i, ch in enumerate(vocab)}
        self.index_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.context_length = context_length

        rng = random.Random(seed)
        scale = 0.1
        self.token_embedding = random_mat(self.vocab_size, embed_dim, scale, rng)
        self.position_embedding = random_mat(context_length, embed_dim, scale, rng)
        self.Wq = random_mat(embed_dim, embed_dim, scale, rng)
        self.Wk = random_mat(embed_dim, embed_dim, scale, rng)
        self.Wv = random_mat(embed_dim, embed_dim, scale, rng)
        self.Wo = random_mat(embed_dim, embed_dim, scale, rng)
        self.W1 = random_mat(embed_dim, ff_dim, scale, rng)
        self.b1 = zeros_vec(ff_dim)
        self.W2 = random_mat(ff_dim, embed_dim, scale, rng)
        self.b2 = zeros_vec(embed_dim)
        self.Wout = random_mat(embed_dim, self.vocab_size, scale, rng)
        self.bout = zeros_vec(self.vocab_size)

        self._param_names = [
            "token_embedding", "position_embedding", "Wq", "Wk", "Wv", "Wo",
            "W1", "b1", "W2", "b2", "Wout", "bout",
        ]
        self._adagrad_memory = {name: self._zeros_like(getattr(self, name)) for name in self._param_names}

    def _zeros_like(self, param):
        if isinstance(param[0], list):
            return zeros_mat(len(param), len(param[0]))
        return zeros_vec(len(param))

    def forward(self, token_indices):
        seq_len = len(token_indices)
        X = [
            [self.token_embedding[idx][d] + self.position_embedding[t][d] for d in range(self.embed_dim)]
            for t, idx in enumerate(token_indices)
        ]

        Q = matmul(X, self.Wq)
        K = matmul(X, self.Wk)
        V = matmul(X, self.Wv)

        scale_factor = 1.0 / math.sqrt(self.embed_dim)
        scores = scale_mat(matmul(Q, transpose(K)), scale_factor)
        mask = causal_mask(seq_len)
        A = softmax_rows(scores, mask=mask)
        O = matmul(A, V)
        attn_out = matmul(O, self.Wo)

        X2 = add_mat(X, attn_out)

        ff_hidden_raw = add_row_broadcast(matmul(X2, self.W1), self.b1)
        ff_hidden = relu_mat(ff_hidden_raw)
        ff_out = add_row_broadcast(matmul(ff_hidden, self.W2), self.b2)

        X3 = add_mat(X2, ff_out)

        logits = add_row_broadcast(matmul(X3, self.Wout), self.bout)

        cache = {
            "token_indices": token_indices, "X": X, "Q": Q, "K": K, "V": V,
            "A": A, "O": O, "attn_out": attn_out, "X2": X2,
            "ff_hidden_raw": ff_hidden_raw, "ff_hidden": ff_hidden, "ff_out": ff_out,
            "X3": X3, "mask": mask, "scale_factor": scale_factor,
        }
        return logits, cache

    def backward(self, dlogits, cache):
        X = cache["X"]
        Q, K, V, A = cache["Q"], cache["K"], cache["V"], cache["A"]
        O, X2 = cache["O"], cache["X2"]
        ff_hidden_raw, ff_hidden, X3 = cache["ff_hidden_raw"], cache["ff_hidden"], cache["X3"]
        mask = cache["mask"]
        scale_factor = cache["scale_factor"]
        seq_len = len(X)

        grads = {}
        grads["Wout"] = matmul(transpose(X3), dlogits)
        grads["bout"] = [sum(dlogits[t][j] for t in range(seq_len)) for j in range(self.vocab_size)]
        dX3 = matmul(dlogits, transpose(self.Wout))

        dX2_from_res = dX3
        dff_out = dX3

        grads["W2"] = matmul(transpose(ff_hidden), dff_out)
        grads["b2"] = [sum(dff_out[t][j] for t in range(seq_len)) for j in range(self.embed_dim)]
        dff_hidden = matmul(dff_out, transpose(self.W2))

        dff_hidden_raw = elementwise_mul(dff_hidden, relu_grad_mat(ff_hidden_raw))
        grads["W1"] = matmul(transpose(X2), dff_hidden_raw)
        grads["b1"] = [sum(dff_hidden_raw[t][j] for t in range(seq_len)) for j in range(self.ff_dim)]
        dX2_from_ff = matmul(dff_hidden_raw, transpose(self.W1))

        dX2 = add_mat(dX2_from_res, dX2_from_ff)

        dattn_out = dX2
        dX_from_res2 = dX2

        grads["Wo"] = matmul(transpose(O), dattn_out)
        dO = matmul(dattn_out, transpose(self.Wo))

        dA = matmul(dO, transpose(V))
        grads["Wv_input"] = None
        dV = matmul(transpose(A), dO)

        dscores = []
        for i in range(seq_len):
            row_A = A[i]
            row_dA = dA[i]
            dot = sum(row_dA[j] * row_A[j] for j in range(seq_len))
            row_dscores = [row_A[j] * (row_dA[j] - dot) if mask[i][j] else 0.0 for j in range(seq_len)]
            dscores.append(row_dscores)

        dQ = scale_mat(matmul(dscores, K), scale_factor)
        dK = scale_mat(matmul(transpose(dscores), Q), scale_factor)

        grads["Wq"] = matmul(transpose(X), dQ)
        grads["Wk"] = matmul(transpose(X), dK)
        grads["Wv"] = matmul(transpose(X), dV)

        dX_from_Q = matmul(dQ, transpose(self.Wq))
        dX_from_K = matmul(dK, transpose(self.Wk))
        dX_from_V = matmul(dV, transpose(self.Wv))

        dX = add_mat(add_mat(dX_from_res2, dX_from_Q), add_mat(dX_from_K, dX_from_V))

        grads["token_embedding"] = zeros_mat(self.vocab_size, self.embed_dim)
        grads["position_embedding"] = zeros_mat(self.context_length, self.embed_dim)
        token_indices = cache["token_indices"]
        for t, idx in enumerate(token_indices):
            for d in range(self.embed_dim):
                grads["token_embedding"][idx][d] += dX[t][d]
                grads["position_embedding"][t][d] += dX[t][d]

        del grads["Wv_input"]
        return grads

    def loss_and_grads(self, input_indices, target_indices):
        logits, cache = self.forward(input_indices)
        seq_len = len(input_indices)
        loss = 0.0
        dlogits = zeros_mat(seq_len, self.vocab_size)
        for t in range(seq_len):
            row = logits[t]
            max_val = max(row)
            exps = [math.exp(v - max_val) for v in row]
            total = sum(exps) or 1e-12
            probs = [e / total for e in exps]
            target = target_indices[t]
            loss += -math.log(max(probs[target], 1e-12))
            for j in range(self.vocab_size):
                dlogits[t][j] = probs[j] / seq_len
            dlogits[t][target] -= 1.0 / seq_len
        grads = self.backward(dlogits, cache)
        return loss / seq_len, grads

    def _clip_and_update(self, grads, learning_rate, clip=5.0):
        for name in self._param_names:
            param = getattr(self, name)
            grad = grads[name]
            memory = self._adagrad_memory[name]
            if isinstance(param[0], list):
                for i in range(len(param)):
                    for j in range(len(param[i])):
                        g = max(-clip, min(clip, grad[i][j]))
                        memory[i][j] += g * g
                        param[i][j] -= learning_rate * g / math.sqrt(memory[i][j] + 1e-8)
            else:
                for i in range(len(param)):
                    g = max(-clip, min(clip, grad[i]))
                    memory[i] += g * g
                    param[i] -= learning_rate * g / math.sqrt(memory[i] + 1e-8)

    def train(self, text, epochs=30, learning_rate=0.05, verbose=False):
        indices = [self.char_to_index[ch] for ch in text if ch in self.char_to_index]
        seq_len = self.context_length
        if len(indices) <= seq_len:
            raise ValueError("training text too short for the given context_length")
        loss_history = []
        for epoch in range(epochs):
            position = 0
            epoch_loss = 0.0
            steps = 0
            while position + seq_len + 1 <= len(indices):
                input_chunk = indices[position:position + seq_len]
                target_chunk = indices[position + 1:position + seq_len + 1]
                loss, grads = self.loss_and_grads(input_chunk, target_chunk)
                self._clip_and_update(grads, learning_rate)
                epoch_loss += loss
                steps += 1
                position += seq_len
            average_loss = epoch_loss / max(1, steps)
            loss_history.append(average_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  loss {average_loss:.4f}")
        return loss_history

    def generate(self, seed_text="", length=100, temperature=0.7, seed=None):
        rng = random.Random(seed)
        context = [self.char_to_index[ch] for ch in seed_text if ch in self.char_to_index]
        if not context:
            context = [rng.randrange(self.vocab_size)]
        output_indices = list(context)

        for _ in range(length):
            window = output_indices[-self.context_length:]
            if len(window) < self.context_length:
                pad = [window[0]] * (self.context_length - len(window))
                window = pad + window
            logits, _ = self.forward(window)
            last_logits = logits[-1]
            scaled = [v / max(temperature, 1e-6) for v in last_logits]
            max_val = max(scaled)
            exps = [math.exp(v - max_val) for v in scaled]
            total = sum(exps) or 1e-12
            probs = [e / total for e in exps]
            r = rng.random()
            cumulative = 0.0
            chosen = len(probs) - 1
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    chosen = i
                    break
            output_indices.append(chosen)

        return "".join(self.index_to_char[i] for i in output_indices)

    def to_dict(self):
        return {name: getattr(self, name) for name in self._param_names} | {
            "vocab": self.vocab, "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim, "context_length": self.context_length,
        }

    @classmethod
    def from_dict(cls, data):
        model = cls(
            vocab=data["vocab"], embed_dim=data["embed_dim"],
            ff_dim=data["ff_dim"], context_length=data["context_length"],
        )
        for name in model._param_names:
            setattr(model, name, data[name])
        return model

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
