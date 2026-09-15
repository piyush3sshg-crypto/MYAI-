from .planner import plan as run_planner


class Method:
    def __init__(self, name, precondition_fn, subtasks_fn):
        self.name = name
        self.precondition_fn = precondition_fn
        self.subtasks_fn = subtasks_fn


class HTNPlanner:
    def __init__(self, action_schemas):
        self.action_schemas_by_name = {schema.name: schema for schema in action_schemas}
        self.methods = {}
        self.decomposition_trace = []

    def add_method(self, task_name, method_name, precondition_fn, subtasks_fn):
        self.methods.setdefault(task_name, []).append(
            Method(method_name, precondition_fn, subtasks_fn)
        )

    def _decompose(self, state, task_list, plan_so_far, depth, max_depth):
        if depth > max_depth:
            return None
        if not task_list:
            return plan_so_far, state

        task_name, args = task_list[0]
        remaining = task_list[1:]

        if task_name in self.action_schemas_by_name:
            schema = self.action_schemas_by_name[task_name]
            binding = dict(zip(schema.parameters, args))
            grounded = schema.ground(binding)
            if not grounded.is_applicable(state):
                self.decomposition_trace.append(
                    f"primitive task {task_name}{args} not applicable, backtracking"
                )
                return None
            new_state = grounded.apply(state)
            self.decomposition_trace.append(f"executed primitive {task_name}{args}")
            return self._decompose(new_state, remaining, plan_so_far + [grounded], depth + 1, max_depth)

        if task_name in self.methods:
            for method in self.methods[task_name]:
                if method.precondition_fn(state, args):
                    subtasks = method.subtasks_fn(args)
                    self.decomposition_trace.append(
                        f"decomposing {task_name}{args} via method {method.name!r} into {subtasks}"
                    )
                    result = self._decompose(state, subtasks + remaining, plan_so_far, depth + 1, max_depth)
                    if result is not None:
                        return result
                    self.decomposition_trace.append(
                        f"method {method.name!r} for {task_name}{args} failed, trying next method"
                    )
            self.decomposition_trace.append(f"no applicable method for {task_name}{args}, backtracking")
            return None

        raise ValueError(f"unknown task or action: {task_name}")

    def plan(self, initial_state, goal_tasks, max_depth=50):
        self.decomposition_trace = []
        result = self._decompose(frozenset(initial_state), goal_tasks, [], 0, max_depth)
        if result is None:
            return None, None
        actions, final_state = result
        return actions, final_state


def validate_plan(initial_state, plan_actions, goal_conditions=None):
    state = frozenset(initial_state)
    for index, action in enumerate(plan_actions):
        if not action.is_applicable(state):
            return {
                "valid": False,
                "failed_at_step": index,
                "failed_action": str(action),
                "state_at_failure": state,
            }
        state = action.apply(state)
    goal_satisfied = True
    if goal_conditions is not None:
        goal_satisfied = frozenset(goal_conditions).issubset(state)
    return {
        "valid": goal_satisfied,
        "failed_at_step": None,
        "final_state": state,
        "goal_satisfied": goal_satisfied,
    }


class RobustPlanExecutor:
    def __init__(self, action_schemas, strategy="astar", max_replans=5):
        self.action_schemas = action_schemas
        self.strategy = strategy
        self.max_replans = max_replans
        self.execution_log = []

    def _log(self, event, **details):
        entry = {"event": event}
        entry.update(details)
        self.execution_log.append(entry)

    def execute(self, initial_state, goal_conditions, disturbance_fn=None):
        self.execution_log = []
        current_state = frozenset(initial_state)

        actions, cost, expansions = run_planner(current_state, goal_conditions, self.action_schemas, self.strategy)
        if actions is None:
            self._log("initial_plan_failed", expansions=expansions)
            return None, self.execution_log
        self._log("initial_plan_found", plan_length=len(actions), cost=cost)

        executed_actions = []
        remaining_actions = list(actions)
        replans_used = 0

        while remaining_actions:
            action = remaining_actions[0]

            if not action.is_applicable(current_state):
                self._log("precondition_failed", action=str(action))
                replans_used += 1
                if replans_used > self.max_replans:
                    self._log("replan_limit_exceeded", limit=self.max_replans)
                    return None, self.execution_log

                new_actions, new_cost, new_expansions = run_planner(
                    current_state, goal_conditions, self.action_schemas, self.strategy
                )
                if new_actions is None:
                    self._log("replan_failed_no_solution")
                    return None, self.execution_log
                self._log("replanned", new_plan_length=len(new_actions), attempt=replans_used)
                remaining_actions = list(new_actions)
                continue

            current_state = action.apply(current_state)
            executed_actions.append(action)
            remaining_actions = remaining_actions[1:]
            self._log("action_executed", action=str(action))

            if disturbance_fn is not None and remaining_actions:
                disturbed_state = disturbance_fn(current_state)
                if frozenset(disturbed_state) != current_state:
                    self._log("disturbance_detected", before=list(current_state), after=list(disturbed_state))
                    current_state = frozenset(disturbed_state)

        goal_reached = frozenset(goal_conditions).issubset(current_state)
        self._log("execution_finished", goal_reached=goal_reached, replans_used=replans_used)
        return executed_actions, self.execution_log
