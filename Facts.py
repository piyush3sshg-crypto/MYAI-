import itertools
from collections import defaultdict


def is_variable(term):
    return isinstance(term, str) and term.startswith("?")


def unify(pattern, fact, bindings=None):
    if bindings is None:
        bindings = {}
    else:
        bindings = dict(bindings)
    if len(pattern) != len(fact):
        return None
    for p_term, f_term in zip(pattern, fact):
        if is_variable(p_term):
            if p_term in bindings:
                if bindings[p_term] != f_term:
                    return None
            else:
                bindings[p_term] = f_term
        else:
            if p_term != f_term:
                return None
    return bindings


def substitute(pattern, bindings):
    return tuple(bindings.get(term, term) if is_variable(term) else term for term in pattern)


class Fact:
    __slots__ = ("predicate", "args", "confidence", "source")

    def __init__(self, predicate, args, confidence=1.0, source=None):
        self.predicate = predicate
        self.args = tuple(args)
        self.confidence = confidence
        self.source = source

    def as_tuple(self):
        return (self.predicate,) + self.args

    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return self.predicate == other.predicate and self.args == other.args

    def __hash__(self):
        return hash((self.predicate, self.args))

    def __repr__(self):
        return f"{self.predicate}({', '.join(map(str, self.args))})"


class FactStore:
    def __init__(self):
        self.by_predicate = defaultdict(set)
        self.by_predicate_arity = defaultdict(set)
        self.all_facts = set()
        self.change_log = []

    def add(self, predicate, args, confidence=1.0, source=None):
        fact = Fact(predicate, args, confidence, source)
        if fact in self.all_facts:
            existing = next(f for f in self.all_facts if f == fact)
            if confidence > existing.confidence:
                self.remove(existing.predicate, existing.args)
            else:
                return fact
        self.all_facts.add(fact)
        self.by_predicate[predicate].add(fact)
        self.by_predicate_arity[(predicate, len(args))].add(fact)
        self.change_log.append(("add", fact))
        return fact

    def remove(self, predicate, args):
        target = Fact(predicate, args)
        if target in self.all_facts:
            self.all_facts.discard(target)
            self.by_predicate[predicate].discard(target)
            self.by_predicate_arity[(predicate, len(args))].discard(target)
            self.change_log.append(("remove", target))
            return True
        return False

    def contains(self, predicate, args):
        return Fact(predicate, args) in self.all_facts

    def query(self, pattern, bindings=None):
        predicate = pattern[0]
        arity = len(pattern) - 1
        candidates = self.by_predicate_arity.get((predicate, arity), set())
        results = []
        for fact in candidates:
            unification = unify(pattern[1:], fact.args, bindings)
            if unification is not None:
                results.append((unification, fact))
        return results

    def query_conjunction(self, patterns):
        solutions = [dict()]
        for pattern in patterns:
            new_solutions = []
            for binding in solutions:
                for new_binding, _ in self.query(pattern, binding):
                    new_solutions.append(new_binding)
            solutions = new_solutions
            if not solutions:
                break
        return solutions

    def all_by_predicate(self, predicate):
        return list(self.by_predicate.get(predicate, set()))

    def predicates(self):
        return list(self.by_predicate.keys())

    def size(self):
        return len(self.all_facts)

    def snapshot(self):
        return set(self.all_facts)

    def restore(self, snapshot):
        self.all_facts = set(snapshot)
        self.by_predicate = defaultdict(set)
        self.by_predicate_arity = defaultdict(set)
        for fact in self.all_facts:
            self.by_predicate[fact.predicate].add(fact)
            self.by_predicate_arity[(fact.predicate, len(fact.args))].add(fact)

    def dump(self):
        return [fact.as_tuple() for fact in self.all_facts]

    def load(self, tuples):
        for t in tuples:
            self.add(t[0], t[1:])


def cartesian_bindings(list_of_binding_lists):
    if not list_of_binding_lists:
        return [{}]
    combined = []
    for combo in itertools.product(*list_of_binding_lists):
        merged = {}
        valid = True
        for binding in combo:
            for k, v in binding.items():
                if k in merged and merged[k] != v:
                    valid = False
                    break
                merged[k] = v
            if not valid:
                break
        if valid:
            combined.append(merged)
    return combined

