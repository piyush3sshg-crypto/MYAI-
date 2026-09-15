from collections import defaultdict, deque


class KnowledgeGraph:
    def __init__(self):
        self.node_attrs = defaultdict(dict)
        self.edges = defaultdict(lambda: defaultdict(set))
        self.reverse_edges = defaultdict(lambda: defaultdict(set))
        self.relation_index = defaultdict(set)
        self.transitive_relations = set()
        self.symmetric_relations = set()

    def add_node(self, node, **attrs):
        self.node_attrs[node].update(attrs)

    def add_edge(self, source, relation, target, weight=1.0):
        self.add_node(source)
        self.add_node(target)
        self.edges[source][relation].add((target, weight))
        self.reverse_edges[target][relation].add((source, weight))
        self.relation_index[relation].add((source, target))
        if relation in self.symmetric_relations:
            self.edges[target][relation].add((source, weight))
            self.reverse_edges[source][relation].add((target, weight))
            self.relation_index[relation].add((target, source))

    def mark_transitive(self, relation):
        self.transitive_relations.add(relation)

    def mark_symmetric(self, relation):
        self.symmetric_relations.add(relation)

    def neighbors(self, node, relation=None):
        if relation is not None:
            return list(self.edges[node].get(relation, set()))
        results = []
        for rel, targets in self.edges[node].items():
            for target, weight in targets:
                results.append((rel, target, weight))
        return results

    def has_edge(self, source, relation, target):
        return (target, ) in [(t,) for t, _ in self.edges[source].get(relation, set())] or any(
            t == target for t, _ in self.edges[source].get(relation, set())
        )

    def query_related(self, node, relation):
        if relation in self.transitive_relations:
            return self._transitive_reach(node, relation)
        return {target for target, _ in self.edges[node].get(relation, set())}

    def _transitive_reach(self, node, relation):
        visited = set()
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for target, _ in self.edges[current].get(relation, set()):
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
        return visited

    def path_exists(self, source, target, relation=None):
        visited = {source}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            if relation is not None:
                candidates = self.edges[current].get(relation, set())
            else:
                candidates = [
                    (t, w) for rel_targets in self.edges[current].values()
                    for t, w in rel_targets
                ]
            for t, _ in candidates:
                if t not in visited:
                    visited.add(t)
                    queue.append(t)
        return source == target

    def shortest_relation_path(self, source, target):
        visited = {source}
        queue = deque([(source, [])])
        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            for rel, targets in self.edges[current].items():
                for t, _ in targets:
                    if t not in visited:
                        visited.add(t)
                        queue.append((t, path + [(rel, t)]))
        return None

    def subgraph(self, nodes):
        nodes = set(nodes)
        sg = KnowledgeGraph()
        for node in nodes:
            sg.add_node(node, **self.node_attrs.get(node, {}))
        for source in nodes:
            for rel, targets in self.edges[source].items():
                for target, weight in targets:
                    if target in nodes:
                        sg.add_edge(source, rel, target, weight)
        return sg

    def infer_transitive_closures(self):
        inferred = []
        for relation in self.transitive_relations:
            pairs = set(self.relation_index.get(relation, set()))
            changed = True
            closure = set(pairs)
            while changed:
                changed = False
                new_pairs = set()
                for a, b in closure:
                    for c, d in closure:
                        if b == c and (a, d) not in closure and a != d:
                            new_pairs.add((a, d))
                if new_pairs:
                    closure |= new_pairs
                    changed = True
            for a, b in closure - pairs:
                self.add_edge(a, relation, b)
                inferred.append((a, relation, b))
        return inferred

    def node_degree(self, node):
        out_degree = sum(len(targets) for targets in self.edges[node].values())
        in_degree = sum(len(sources) for sources in self.reverse_edges[node].values())
        return in_degree, out_degree

    def most_connected_nodes(self, top_k=10):
        scores = []
        all_nodes = set(self.node_attrs.keys())
        for node in all_nodes:
            in_deg, out_deg = self.node_degree(node)
            scores.append((node, in_deg + out_deg))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_fact_tuples(self):
        tuples = []
        for source, relations in self.edges.items():
            for relation, targets in relations.items():
                for target, weight in targets:
                    tuples.append((relation, source, target, weight))
        return tuples

    def load_from_facts(self, fact_store, relation_predicates):
        for predicate in relation_predicates:
            for fact in fact_store.all_by_predicate(predicate):
                if len(fact.args) >= 2:
                    self.add_edge(fact.args[0], predicate, fact.args[1], fact.confidence)


class ConceptHierarchy:
    def __init__(self):
        self.graph = KnowledgeGraph()
        self.graph.mark_transitive("is_a")

    def add_concept(self, concept, parent=None):
        self.graph.add_node(concept)
        if parent is not None:
            self.graph.add_edge(concept, "is_a", parent)

    def is_ancestor(self, concept, ancestor):
        return ancestor in self.graph.query_related(concept, "is_a")

    def ancestors(self, concept):
        return self.graph.query_related(concept, "is_a")

    def common_ancestor(self, concept_a, concept_b):
        ancestors_a = self.ancestors(concept_a) | {concept_a}
        ancestors_b = self.ancestors(concept_b) | {concept_b}
        shared = ancestors_a & ancestors_b
        if not shared:
            return None
        best = None
        best_depth = -1
        for candidate in shared:
            depth = len(self.ancestors(candidate))
            if depth > best_depth:
                best_depth = depth
                best = candidate
        return best

