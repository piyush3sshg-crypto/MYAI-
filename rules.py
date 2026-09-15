import itertools
from Facts import unify, substitute, is_variable


class Rule:
    def __init__(self, name, conditions, conclusions, priority=0, confidence_fn=None):
        self.name = name
        self.conditions = conditions
        self.conclusions = conclusions
        self.priority = priority
        self.confidence_fn = confidence_fn or (lambda confs: min(confs) if confs else 1.0)
        self.fire_count = 0

    def specificity(self):
        return sum(len(c) for c in self.conditions)

    def __repr__(self):
        return f"Rule({self.name})"


class RuleFiring:
    def __init__(self, rule, bindings, produced_facts):
        self.rule = rule
        self.bindings = bindings
        self.produced_facts = produced_facts


class RuleEngine:
    def __init__(self, fact_store):
        self.fact_store = fact_store
        self.rules = []
        self.firing_history = []
        self.agenda_trace = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def add_rules(self, rules):
        self.rules.extend(rules)

    def _match_rule(self, rule):
        solutions = self.fact_store.query_conjunction(rule.conditions)
        return solutions

    def _instantiate_conclusions(self, rule, bindings):
        produced = []
        for conclusion in rule.conclusions:
            predicate = conclusion[0]
            args = substitute(conclusion[1:], bindings)
            if any(is_variable(a) for a in args):
                continue
            produced.append((predicate, args))
        return produced

    def forward_chain(self, max_iterations=1000):
        total_new_facts = 0
        for iteration in range(max_iterations):
            candidate_firings = []
            for rule in self.rules:
                solutions = self._match_rule(rule)
                for bindings in solutions:
                    produced = self._instantiate_conclusions(rule, bindings)
                    new_produced = [
                        (p, a) for p, a in produced
                        if not self.fact_store.contains(p, a)
                    ]
                    if new_produced:
                        candidate_firings.append((rule, bindings, new_produced))
            if not candidate_firings:
                break
            candidate_firings.sort(key=lambda cf: (-cf[0].priority, -cf[0].specificity()))
            new_facts_this_round = 0
            for rule, bindings, produced in candidate_firings:
                actually_new = [
                    (p, a) for p, a in produced
                    if not self.fact_store.contains(p, a)
                ]
                if not actually_new:
                    continue
                for predicate, args in actually_new:
                    self.fact_store.add(predicate, args, source=rule.name)
                rule.fire_count += 1
                self.firing_history.append(RuleFiring(rule, bindings, actually_new))
                new_facts_this_round += len(actually_new)
            total_new_facts += new_facts_this_round
            self.agenda_trace.append(new_facts_this_round)
            if new_facts_this_round == 0:
                break
        return total_new_facts

    def backward_chain(self, goal, bindings=None, depth=0, max_depth=50, visited=None):
        if bindings is None:
            bindings = {}
        if visited is None:
            visited = set()
        if depth > max_depth:
            return []
        instantiated_goal = substitute(goal, bindings)
        if instantiated_goal in visited:
            return []
        visited = visited | {instantiated_goal}
        direct_matches = self.fact_store.query(goal, bindings)
        results = [b for b, _ in direct_matches]
        for rule in self.rules:
            for conclusion in rule.conclusions:
                if conclusion[0] != goal[0]:
                    continue
                conclusion_bindings = unify(conclusion[1:], instantiated_goal[1:], {})
                if conclusion_bindings is None:
                    continue
                merged_bindings = dict(bindings)
                consistent = True
                for k, v in conclusion_bindings.items():
                    if k in merged_bindings and merged_bindings[k] != v:
                        consistent = False
                        break
                    merged_bindings[k] = v
                if not consistent:
                    continue
                sub_solution_sets = []
                for condition in rule.conditions:
                    sub_results = self.backward_chain(
                        condition, merged_bindings, depth + 1, max_depth, visited
                    )
                    if not sub_results:
                        sub_solution_sets = []
                        break
                    sub_solution_sets.append(sub_results)
                if not sub_solution_sets and rule.conditions:
                    continue
                if not rule.conditions:
                    results.append(merged_bindings)
                    continue
                for combo in itertools.product(*sub_solution_sets):
                    combined = dict(merged_bindings)
                    ok = True
                    for b in combo:
                        for k, v in b.items():
                            if k in combined and combined[k] != v:
                                ok = False
                                break
                            combined[k] = v
                        if not ok:
                            break
                    if ok:
                        results.append(combined)
        unique_results = []
        seen = set()
        for r in results:
            key = tuple(sorted(r.items()))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        return unique_results

    def explain(self, predicate, args):
        for firing in reversed(self.firing_history):
            for p, a in firing.produced_facts:
                if p == predicate and a == args:
                    return firing
        return None

    def reset_history(self):
        self.firing_history = []
        self.agenda_trace = []
        for rule in self.rules:
            rule.fire_count = 0

