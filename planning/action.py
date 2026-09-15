import itertools
from Facts import is_variable, unify, substitute


class ActionSchema:
    def __init__(self, name, parameters, preconditions, add_effects, del_effects, cost=1.0):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.add_effects = add_effects
        self.del_effects = del_effects
        self.cost = cost

    def ground(self, binding):
        grounded_preconditions = [
            (p[0],) + substitute(p[1:], binding) for p in self.preconditions
        ]
        grounded_add = [(a[0],) + substitute(a[1:], binding) for a in self.add_effects]
        grounded_del = [(d[0],) + substitute(d[1:], binding) for d in self.del_effects]
        args = tuple(binding.get(p, p) for p in self.parameters)
        return GroundedAction(
            f"{self.name}({', '.join(map(str, args))})",
            grounded_preconditions,
            grounded_add,
            grounded_del,
            self.cost,
        )

    def possible_groundings(self, state_facts, object_pool):
        candidate_bindings = [dict()]
        for param in self.parameters:
            new_candidates = []
            for binding in candidate_bindings:
                if param in binding:
                    new_candidates.append(binding)
                    continue
                for obj in object_pool:
                    new_binding = dict(binding)
                    new_binding[param] = obj
                    new_candidates.append(new_binding)
            candidate_bindings = new_candidates
        groundings = []
        for binding in candidate_bindings:
            grounded = self.ground(binding)
            if grounded.is_applicable(state_facts):
                groundings.append(grounded)
        return groundings


class GroundedAction:
    def __init__(self, name, preconditions, add_effects, del_effects, cost=1.0):
        self.name = name
        self.preconditions = frozenset(preconditions)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)
        self.cost = cost

    def is_applicable(self, state_facts):
        return self.preconditions.issubset(state_facts)

    def apply(self, state_facts):
        new_state = set(state_facts)
        new_state -= self.del_effects
        new_state |= self.add_effects
        return frozenset(new_state)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, GroundedAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


def state_from_fact_store(fact_store, relevant_predicates=None):
    facts = fact_store.dump()
    if relevant_predicates is not None:
        facts = [f for f in facts if f[0] in relevant_predicates]
    return frozenset(facts)


def extract_objects(state_facts):
    objects = set()
    for fact in state_facts:
        for term in fact[1:]:
            objects.add(term)
    return objects

