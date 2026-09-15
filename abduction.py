from Facts import unify, substitute, is_variable


class Hypothesis:
    def __init__(self, rule, bindings, required_facts, plausibility):
        self.rule = rule
        self.bindings = bindings
        self.required_facts = required_facts
        self.plausibility = plausibility

    def __repr__(self):
        return f"Hypothesis(rule={self.rule.name}, needs={self.required_facts}, p={self.plausibility:.3f})"


class AbductiveReasoner:
    def __init__(self, fact_store, rule_engine):
        self.fact_store = fact_store
        self.rule_engine = rule_engine

    def explain_observation(self, observation, max_hypotheses=10):
        predicate = observation[0]
        hypotheses = []
        for rule in self.rule_engine.rules:
            for conclusion in rule.conclusions:
                if conclusion[0] != predicate:
                    continue
                bindings = unify(conclusion[1:], observation[1:], {})
                if bindings is None:
                    continue
                required_facts = []
                satisfied_count = 0
                for condition in rule.conditions:
                    grounded_condition = substitute(condition, bindings)
                    if any(is_variable(t) for t in grounded_condition):
                        required_facts.append(grounded_condition)
                        continue
                    if self.fact_store.contains(grounded_condition[0], grounded_condition[1:]):
                        satisfied_count += 1
                    else:
                        required_facts.append(grounded_condition)
                total_conditions = len(rule.conditions) or 1
                plausibility = satisfied_count / total_conditions
                hypotheses.append(Hypothesis(rule, bindings, required_facts, plausibility))
        hypotheses.sort(key=lambda h: (-h.plausibility, len(h.required_facts)))
        return hypotheses[:max_hypotheses]

    def best_explanation(self, observation):
        hypotheses = self.explain_observation(observation, max_hypotheses=1)
        if not hypotheses:
            return None
        return hypotheses[0]

    def minimal_covering_hypotheses(self, observations):
        all_hypotheses = {}
        for obs in observations:
            all_hypotheses[obs] = self.explain_observation(obs)
        combined_facts_needed = set()
        chosen = {}
        for obs, hyps in all_hypotheses.items():
            if hyps:
                chosen[obs] = hyps[0]
                combined_facts_needed.update(tuple(f) for f in hyps[0].required_facts)
        return chosen, combined_facts_needed

    def rank_hypotheses_by_coverage(self, observations):
        rule_coverage = {}
        for obs in observations:
            for hyp in self.explain_observation(obs, max_hypotheses=100):
                key = hyp.rule.name
                rule_coverage.setdefault(key, {"rule": hyp.rule, "covers": set(), "hypotheses": []})
                rule_coverage[key]["covers"].add(obs)
                rule_coverage[key]["hypotheses"].append(hyp)
        ranked = sorted(rule_coverage.values(), key=lambda r: len(r["covers"]), reverse=True)
        return ranked

