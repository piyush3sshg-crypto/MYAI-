import time
from collections import defaultdict


class ActivationUnit:
    __slots__ = ("content", "activation", "last_updated", "links")

    def __init__(self, content, activation=1.0):
        self.content = content
        self.activation = activation
        self.last_updated = time.time()
        self.links = {}


class WorkingMemory:
    def __init__(self, decay_rate=0.05, activation_threshold=0.1, capacity=200):
        self.units = {}
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.capacity = capacity
        self.tick_count = 0

    def _key(self, content):
        return content if isinstance(content, (str, tuple)) else repr(content)

    def add(self, content, initial_activation=1.0):
        key = self._key(content)
        if key in self.units:
            self.units[key].activation = min(1.0, self.units[key].activation + initial_activation)
        else:
            self.units[key] = ActivationUnit(content, initial_activation)
        if len(self.units) > self.capacity:
            self._prune_weakest()
        return key

    def link(self, content_a, content_b, weight=0.5):
        key_a, key_b = self._key(content_a), self._key(content_b)
        if key_a in self.units and key_b in self.units:
            self.units[key_a].links[key_b] = weight
            self.units[key_b].links[key_a] = weight

    def activate(self, content, amount=0.5):
        key = self._key(content)
        if key not in self.units:
            self.add(content, amount)
            return
        self.units[key].activation = min(1.0, self.units[key].activation + amount)
        self.units[key].last_updated = time.time()
        self._spread(key, amount * 0.5, depth=2)

    def _spread(self, key, amount, depth):
        if depth <= 0 or amount < 0.01:
            return
        unit = self.units.get(key)
        if unit is None:
            return
        for neighbor_key, weight in unit.links.items():
            neighbor = self.units.get(neighbor_key)
            if neighbor is None:
                continue
            transferred = amount * weight
            neighbor.activation = min(1.0, neighbor.activation + transferred)
            self._spread(neighbor_key, transferred * 0.5, depth - 1)

    def decay(self):
        self.tick_count += 1
        to_remove = []
        for key, unit in self.units.items():
            unit.activation *= (1.0 - self.decay_rate)
            if unit.activation < self.activation_threshold:
                to_remove.append(key)
        for key in to_remove:
            del self.units[key]

    def _prune_weakest(self):
        if not self.units:
            return
        weakest_key = min(self.units, key=lambda k: self.units[k].activation)
        del self.units[weakest_key]

    def top_active(self, n=10):
        ordered = sorted(self.units.values(), key=lambda u: u.activation, reverse=True)
        return [(u.content, u.activation) for u in ordered[:n]]

    def contains(self, content):
        return self._key(content) in self.units

    def get_activation(self, content):
        key = self._key(content)
        if key in self.units:
            return self.units[key].activation
        return 0.0

    def clear(self):
        self.units.clear()
        self.tick_count = 0


class AttentionFocus:
    def __init__(self, working_memory, focus_size=5):
        self.working_memory = working_memory
        self.focus_size = focus_size
        self.history = []

    def compute_focus(self):
        top = self.working_memory.top_active(self.focus_size)
        self.history.append([content for content, _ in top])
        if len(self.history) > 100:
            self.history.pop(0)
        return top

    def focus_stability(self):
        if len(self.history) < 2:
            return 1.0
        overlaps = []
        for i in range(1, len(self.history)):
            prev_set = set(self.history[i - 1])
            cur_set = set(self.history[i])
            if not prev_set and not cur_set:
                overlaps.append(1.0)
                continue
            union = prev_set | cur_set
            overlaps.append(len(prev_set & cur_set) / len(union) if union else 1.0)
        return sum(overlaps) / len(overlaps)

