import heapq
import itertools
from collections import OrderedDict, deque, defaultdict


class TrieNode:
    __slots__ = ("children", "is_terminal", "payload")

    def __init__(self):
        self.children = {}
        self.is_terminal = False
        self.payload = None


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, key, payload=None):
        node = self.root
        for ch in key:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if not node.is_terminal:
            self.size += 1
        node.is_terminal = True
        node.payload = payload

    def search(self, key):
        node = self._walk(key)
        if node is None or not node.is_terminal:
            return None
        return node.payload

    def starts_with(self, prefix):
        node = self._walk(prefix)
        if node is None:
            return []
        results = []
        stack = [(prefix, node)]
        while stack:
            cur_prefix, cur_node = stack.pop()
            if cur_node.is_terminal:
                results.append((cur_prefix, cur_node.payload))
            for ch, child in cur_node.children.items():
                stack.append((cur_prefix + ch, child))
        return results

    def delete(self, key):
        node = self._walk(key)
        if node is None or not node.is_terminal:
            return False
        node.is_terminal = False
        node.payload = None
        self.size -= 1
        return True

    def _walk(self, key):
        node = self.root
        for ch in key:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node


class UpdatablePriorityQueue:
    REMOVED = object()

    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def push(self, item, priority):
        if item in self.entry_finder:
            self.remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)

    def remove(self, item):
        entry = self.entry_finder.pop(item, None)
        if entry is not None:
            entry[2] = UpdatablePriorityQueue.REMOVED

    def pop(self):
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            if item is not UpdatablePriorityQueue.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError("pop from an empty priority queue")

    def is_empty(self):
        return not self.entry_finder

    def __len__(self):
        return len(self.entry_finder)


class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self.make_set(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def groups(self):
        buckets = defaultdict(list)
        for item in self.parent:
            buckets[self.find(item)].append(item)
        return list(buckets.values())


class WeightedGraph:
    def __init__(self, directed=True):
        self.directed = directed
        self.adjacency = defaultdict(dict)
        self.nodes = set()

    def add_node(self, node, **attrs):
        self.nodes.add(node)
        if node not in self.adjacency:
            self.adjacency[node] = {}
        return attrs

    def add_edge(self, u, v, weight=1.0):
        self.add_node(u)
        self.add_node(v)
        self.adjacency[u][v] = weight
        if not self.directed:
            self.adjacency[v][u] = weight

    def remove_edge(self, u, v):
        if v in self.adjacency.get(u, {}):
            del self.adjacency[u][v]
        if not self.directed and u in self.adjacency.get(v, {}):
            del self.adjacency[v][u]

    def neighbors(self, node):
        return list(self.adjacency.get(node, {}).items())

    def bfs(self, start):
        visited = {start}
        order = []
        queue = deque([start])
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in sorted(self.neighbors(node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order

    def dfs(self, start):
        visited = set()
        order = []

        def _visit(node):
            visited.add(node)
            order.append(node)
            for neighbor, _ in sorted(self.neighbors(node)):
                if neighbor not in visited:
                    _visit(neighbor)

        _visit(start)
        return order

    def dijkstra(self, start):
        distances = {node: float("inf") for node in self.nodes}
        previous = {node: None for node in self.nodes}
        distances[start] = 0
        pq = UpdatablePriorityQueue()
        for node in self.nodes:
            pq.push(node, distances[node])
        visited = set()
        while not pq.is_empty():
            try:
                node, dist = pq.pop()
            except KeyError:
                break
            if node in visited:
                continue
            visited.add(node)
            for neighbor, weight in self.neighbors(node):
                if neighbor in visited:
                    continue
                candidate = dist + weight
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = node
                    pq.push(neighbor, candidate)
        return distances, previous

    def shortest_path(self, start, end):
        distances, previous = self.dijkstra(start)
        if distances.get(end, float("inf")) == float("inf"):
            return None, float("inf")
        path = []
        node = end
        while node is not None:
            path.append(node)
            node = previous[node]
        path.reverse()
        return path, distances[end]

    def topological_sort(self):
        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for neighbor, _ in self.neighbors(node):
                in_degree[neighbor] += 1
        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.neighbors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.nodes):
            return None
        return order

    def transitive_closure(self):
        closure = {node: set(self.bfs(node)) - {node} for node in self.nodes}
        return closure

    def connected_components(self):
        ds = DisjointSet()
        for node in self.nodes:
            ds.make_set(node)
        for u in self.nodes:
            for v, _ in self.neighbors(u):
                ds.union(u, v)
        return ds.groups()


class LRUCache:
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.store = OrderedDict()

    def get(self, key, default=None):
        if key not in self.store:
            return default
        self.store.move_to_end(key)
        return self.store[key]

    def put(self, key, value):
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)


class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, item):
        self.buffer.append(item)

    def to_list(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)


class MinMaxHeap:
    def __init__(self):
        self._min_heap = []
        self._max_heap = []
        self._counter = itertools.count()
        self._active = {}

    def push(self, value):
        count = next(self._counter)
        self._active[count] = value
        heapq.heappush(self._min_heap, (value, count))
        heapq.heappush(self._max_heap, (-value, count))
        return count

    def _prune(self, heap, sign):
        while heap:
            value, count = heap[0]
            if count in self._active and self._active[count] == sign * value:
                return
            heapq.heappop(heap)

    def peek_min(self):
        self._prune(self._min_heap, 1)
        if not self._min_heap:
            return None
        return self._min_heap[0][0]

    def peek_max(self):
        self._prune(self._max_heap, -1)
        if not self._max_heap:
            return None
        return -self._max_heap[0][0]

    def pop_min(self):
        self._prune(self._min_heap, 1)
        if not self._min_heap:
            return None
        value, count = heapq.heappop(self._min_heap)
        del self._active[count]
        return value

    def pop_max(self):
        self._prune(self._max_heap, -1)
        if not self._max_heap:
            return None
        neg_value, count = heapq.heappop(self._max_heap)
        del self._active[count]
        return -neg_value

