from collections import defaultdict, Counter
from rules import Rule


class TrainingExample:
    def __init__(self, attributes, label):
        self.attributes = attributes
        self.label = label


class VersionSpaceLearner:
    def __init__(self, attribute_domains):
        self.attribute_domains = attribute_domains
        self.general_boundary = [self._most_general()]
        self.specific_boundary = []

    def _most_general(self):
        return {attr: "?" for attr in self.attribute_domains}

    def _most_specific(self, example):
        return dict(example.attributes)

    def _is_consistent(self, hypothesis, example):
        for attr, value in hypothesis.items():
            if value == "?":
                continue
            if value == "0":
                return False
            if example.attributes.get(attr) != value:
                return False
        return True

    def _generalize(self, hypothesis, example):
        new_hyp = dict(hypothesis)
        for attr in self.attribute_domains:
            if new_hyp.get(attr) == "0":
                new_hyp[attr] = example.attributes.get(attr, "?")
            elif new_hyp.get(attr) != example.attributes.get(attr):
                new_hyp[attr] = "?"
        return new_hyp

    def _specialize_candidates(self, hypothesis, example):
        candidates = []
        for attr, domain in self.attribute_domains.items():
            if hypothesis.get(attr) == "?":
                for value in domain:
                    if value != example.attributes.get(attr):
                        new_hyp = dict(hypothesis)
                        new_hyp[attr] = value
                        candidates.append(new_hyp)
        return candidates

    def train(self, examples):
        self.specific_boundary = [{attr: "0" for attr in self.attribute_domains}]
        for example in examples:
            if example.label:
                self.general_boundary = [
                    g for g in self.general_boundary if self._is_consistent(g, example)
                ]
                new_specific = []
                for s in self.specific_boundary:
                    if self._is_consistent(s, example):
                        new_specific.append(s)
                    else:
                        new_specific.append(self._generalize(s, example))
                self.specific_boundary = self._remove_more_general_duplicates(new_specific)
            else:
                new_general = []
                for g in self.general_boundary:
                    if not self._is_consistent(g, example):
                        new_general.append(g)
                    else:
                        new_general.extend(self._specialize_candidates(g, example))
                self.general_boundary = self._remove_more_specific_duplicates(new_general)
        return self.specific_boundary, self.general_boundary

    def _remove_more_general_duplicates(self, hyps):
        unique = []
        seen = set()
        for h in hyps:
            key = tuple(sorted(h.items()))
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique

    def _remove_more_specific_duplicates(self, hyps):
        return self._remove_more_general_duplicates(hyps)


class DecisionStumpLearner:
    def __init__(self):
        self.best_attribute = None
        self.best_split = None
        self.branches = {}

    def _entropy(self, examples):
        if not examples:
            return 0.0
        counts = Counter(e.label for e in examples)
        total = len(examples)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * (p.bit_length() if False else __import__("math").log2(p))
        return entropy

    def _information_gain(self, examples, attribute):
        base_entropy = self._entropy(examples)
        groups = defaultdict(list)
        for e in examples:
            groups[e.attributes.get(attribute)].append(e)
        weighted_entropy = 0.0
        total = len(examples)
        for group in groups.values():
            weighted_entropy += (len(group) / total) * self._entropy(group)
        return base_entropy - weighted_entropy

    def train(self, examples, candidate_attributes):
        best_gain = -1
        best_attr = None
        for attr in candidate_attributes:
            gain = self._information_gain(examples, attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        self.best_attribute = best_attr
        if best_attr is not None:
            groups = defaultdict(list)
            for e in examples:
                groups[e.attributes.get(best_attr)].append(e)
            for value, group in groups.items():
                majority_label = Counter(e.label for e in group).most_common(1)[0][0]
                self.branches[value] = majority_label
        return self.best_attribute, self.branches

    def predict(self, attributes):
        if self.best_attribute is None:
            return None
        value = attributes.get(self.best_attribute)
        return self.branches.get(value)


class RuleInducer:
    def __init__(self, fact_store, target_predicate, candidate_predicates):
        self.fact_store = fact_store
        self.target_predicate = target_predicate
        self.candidate_predicates = candidate_predicates

    def induce_rules_from_examples(self, positive_examples, negative_examples, max_conditions=3):
        common_conditions_sets = []
        for pos_example in positive_examples:
            related_facts = self._related_facts(pos_example)
            common_conditions_sets.append(set(related_facts))
        if not common_conditions_sets:
            return []
        shared = set.intersection(*common_conditions_sets) if common_conditions_sets else set()
        induced_conditions = list(shared)[:max_conditions]
        if not induced_conditions:
            return []
        variable_map = {}
        conditions = []
        for fact_tuple in induced_conditions:
            predicate = fact_tuple[0]
            args = []
            for arg in fact_tuple[1:]:
                if arg not in variable_map:
                    variable_map[arg] = f"?v{len(variable_map)}"
                args.append(variable_map[arg])
            conditions.append((predicate,) + tuple(args))
        target_args = tuple(
            variable_map.get(a, a) for a in positive_examples[0]
        )
        rule = Rule(
            name=f"induced_{self.target_predicate}",
            conditions=conditions,
            conclusions=[(self.target_predicate,) + target_args],
        )
        return [rule]

    def _related_facts(self, entity_tuple):
        related = []
        for predicate in self.candidate_predicates:
            for fact in self.fact_store.all_by_predicate(predicate):
                if any(arg in entity_tuple for arg in fact.args):
                    related.append(fact.as_tuple())
        return related

