from Facts import is_variable


class DeductionResult:
    def __init__(self, query, bindings_list, proof_confidence, source):
        self.query = query
        self.bindings_list = bindings_list
        self.proof_confidence = proof_confidence
        self.source = source

    def is_satisfied(self):
        return len(self.bindings_list) > 0

    def distinct_answers(self, variable):
        answers = set()
        for binding in self.bindings_list:
            if variable in binding:
                answers.add(binding[variable])
        return answers


class DeductiveReasoner:
    def __init__(self, fact_store, rule_engine, knowledge_graph=None):
        self.fact_store = fact_store
        self.rule_engine = rule_engine
        self.knowledge_graph = knowledge_graph

    def answer_query(self, query, use_forward_chaining=True):
        if use_forward_chaining:
            self.rule_engine.forward_chain()
        direct = self.fact_store.query(query)
        bindings_list = [b for b, _ in direct]
        if not bindings_list:
            bindings_list = self.rule_engine.backward_chain(query)
        confidence = 1.0 if bindings_list else 0.0
        source = "fact_store" if direct else "rule_engine"
        return DeductionResult(query, bindings_list, confidence, source)

    def answer_conjunctive_query(self, patterns):
        self.rule_engine.forward_chain()
        return self.fact_store.query_conjunction(patterns)

    def check_relation_via_graph(self, source_node, relation, target_node):
        if self.knowledge_graph is None:
            return False
        related = self.knowledge_graph.query_related(source_node, relation)
        return target_node in related

    def syllogism(self, major_premise, minor_premise):
        major_pred, major_a, major_b = major_premise
        minor_pred, minor_a, minor_b = minor_premise
        if major_a == minor_b:
            return (major_pred, minor_a, major_b)
        if major_b == minor_a:
            return (minor_pred, major_a, minor_b)
        return None

    def chain_inference(self, start_predicate, start_args, chain_predicates):
        current_facts = self.fact_store.query((start_predicate,) + tuple(start_args))
        results = [b for b, _ in current_facts]
        for predicate in chain_predicates:
            next_results = []
            for binding in results:
                pattern = (predicate,) + tuple(f"?x{i}" for i in range(2))
                matches = self.fact_store.query(pattern, binding)
                next_results.extend(b for b, _ in matches)
            results = next_results
        return results

    def confidence_weighted_answer(self, predicate, args):
        matches = [f for f in self.fact_store.all_by_predicate(predicate) if f.args == tuple(args)]
        if not matches:
            return 0.0
        return max(f.confidence for f in matches)

