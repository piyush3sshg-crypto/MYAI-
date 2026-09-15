from Datastructures import UpdatablePriorityQueue
from .action import extract_objects


class PlanningProblem:
    def __init__(self, initial_state, goal_conditions, action_schemas):
        self.initial_state = frozenset(initial_state)
        self.goal_conditions = frozenset(goal_conditions)
        self.action_schemas = action_schemas
        self.object_pool = extract_objects(self.initial_state)

    def is_goal(self, state):
        return self.goal_conditions.issubset(state)

    def applicable_actions(self, state):
        actions = []
        for schema in self.action_schemas:
            actions.extend(schema.possible_groundings(state, self.object_pool))
        return actions


def goal_count_heuristic(state, goal_conditions):
    return len(goal_conditions - state)


def relaxed_plan_heuristic(state, goal_conditions, action_schemas, object_pool, max_layers=25):
    if goal_conditions.issubset(state):
        return 0
    layers = [set(state)]
    achieved = set(state)
    action_layers = []
    for _ in range(max_layers):
        applicable = []
        for schema in action_schemas:
            applicable.extend(schema.possible_groundings(achieved, object_pool))
        new_facts = set()
        layer_actions = []
        for action in applicable:
            if not action.add_effects.issubset(achieved):
                new_facts |= action.add_effects
                layer_actions.append(action)
        if not new_facts - achieved:
            break
        achieved |= new_facts
        action_layers.append(layer_actions)
        if goal_conditions.issubset(achieved):
            return len(action_layers)
    if goal_conditions.issubset(achieved):
        return len(action_layers)
    return len(action_layers) + len(goal_conditions - achieved) * 2


class PlanNode:
    __slots__ = ("state", "parent", "action", "g", "h")

    def __init__(self, state, parent, action, g, h):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h

    def f(self):
        return self.g + self.h

    def reconstruct(self):
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions


class AStarPlanner:
    def __init__(self, problem, heuristic="relaxed", max_expansions=20000):
        self.problem = problem
        self.heuristic_mode = heuristic
        self.max_expansions = max_expansions

    def _heuristic(self, state):
        if self.heuristic_mode == "goal_count":
            return goal_count_heuristic(state, self.problem.goal_conditions)
        return relaxed_plan_heuristic(
            state,
            self.problem.goal_conditions,
            self.problem.action_schemas,
            self.problem.object_pool,
        )

    def search(self):
        start_state = self.problem.initial_state
        start_h = self._heuristic(start_state)
        start_node = PlanNode(start_state, None, None, 0, start_h)
        open_set = UpdatablePriorityQueue()
        open_set.push(start_state, start_node.f())
        node_registry = {start_state: start_node}
        closed = set()
        expansions = 0
        while not open_set.is_empty() and expansions < self.max_expansions:
            current_state, _ = open_set.pop()
            current_node = node_registry[current_state]
            if self.problem.is_goal(current_state):
                return current_node.reconstruct(), current_node.g, expansions
            if current_state in closed:
                continue
            closed.add(current_state)
            expansions += 1
            for action in self.problem.applicable_actions(current_state):
                next_state = action.apply(current_state)
                tentative_g = current_node.g + action.cost
                existing_node = node_registry.get(next_state)
                if existing_node is None or tentative_g < existing_node.g:
                    h = self._heuristic(next_state)
                    new_node = PlanNode(next_state, current_node, action, tentative_g, h)
                    node_registry[next_state] = new_node
                    if next_state not in closed:
                        open_set.push(next_state, new_node.f())
        return None, float("inf"), expansions


class GreedyBestFirstPlanner:
    def __init__(self, problem, max_expansions=20000):
        self.problem = problem
        self.max_expansions = max_expansions

    def search(self):
        start_state = self.problem.initial_state
        start_node = PlanNode(start_state, None, None, 0, goal_count_heuristic(start_state, self.problem.goal_conditions))
        open_set = UpdatablePriorityQueue()
        open_set.push(start_state, start_node.h)
        node_registry = {start_state: start_node}
        visited = set()
        expansions = 0
        while not open_set.is_empty() and expansions < self.max_expansions:
            current_state, _ = open_set.pop()
            current_node = node_registry[current_state]
            if self.problem.is_goal(current_state):
                return current_node.reconstruct(), current_node.g, expansions
            if current_state in visited:
                continue
            visited.add(current_state)
            expansions += 1
            for action in self.problem.applicable_actions(current_state):
                next_state = action.apply(current_state)
                if next_state in visited:
                    continue
                h = goal_count_heuristic(next_state, self.problem.goal_conditions)
                new_node = PlanNode(next_state, current_node, action, current_node.g + action.cost, h)
                if next_state not in node_registry or new_node.g < node_registry[next_state].g:
                    node_registry[next_state] = new_node
                    open_set.push(next_state, h)
        return None, float("inf"), expansions


def plan(initial_state, goal_conditions, action_schemas, strategy="astar"):
    problem = PlanningProblem(initial_state, goal_conditions, action_schemas)
    if strategy == "greedy":
        planner = GreedyBestFirstPlanner(problem)
    else:
        planner = AStarPlanner(problem)
    return planner.search()

