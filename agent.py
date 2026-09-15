import time


class AutonomousAgent:
    def __init__(self, name="Agent", tool_agent=None, plan_executor=None, max_steps=20, max_consecutive_failures=3):
        self.name = name
        self.tool_agent = tool_agent
        self.plan_executor = plan_executor
        self.max_steps = max_steps
        self.max_consecutive_failures = max_consecutive_failures
        self.memory = []
        self.goal_queue = []
        self.stopped_reason = None

    def add_tool_goal(self, text):
        self.goal_queue.append({"kind": "tool_request", "payload": text, "status": "pending", "result": None})

    def add_plan_goal(self, initial_state, goal_conditions):
        self.goal_queue.append({
            "kind": "plan_goal",
            "payload": (initial_state, goal_conditions),
            "status": "pending",
            "result": None,
        })

    def _log(self, event, **details):
        entry = {"event": event, "timestamp": time.time()}
        entry.update(details)
        self.memory.append(entry)

    def run(self):
        self.stopped_reason = None
        steps_taken = 0
        consecutive_failures = 0

        for goal in self.goal_queue:
            if goal["status"] != "pending":
                continue

            if steps_taken >= self.max_steps:
                self.stopped_reason = "max_steps_exceeded"
                self._log("stopped", reason=self.stopped_reason, steps_taken=steps_taken)
                break

            self._log("perceive_goal", kind=goal["kind"], payload=str(goal["payload"])[:120])

            success = False

            if goal["kind"] == "tool_request":
                if self.tool_agent is None:
                    self._log("failed", reason="no tool agent configured")
                else:
                    result = self.tool_agent.run(goal["payload"])
                    self._log(
                        "act", mechanism="tool", tool=result["tool_selected"],
                        confidence=result["confidence"], result=str(result["result"])[:200],
                    )
                    if result.get("error") is None and result["tool_selected"] != "unknown":
                        success = True
                        goal["result"] = result["result"]

            elif goal["kind"] == "plan_goal":
                if self.plan_executor is None:
                    self._log("failed", reason="no plan executor configured")
                else:
                    initial_state, goal_conditions = goal["payload"]
                    executed, execution_log = self.plan_executor.execute(initial_state, goal_conditions)
                    for entry in execution_log:
                        sub_event = entry.get("event")
                        entry_details = {k: v for k, v in entry.items() if k != "event"}
                        self._log("plan_step", mechanism="planner", sub_event=sub_event, **entry_details)
                    if executed is not None:
                        success = True
                        goal["result"] = [str(a) for a in executed]

            else:
                self._log("failed", reason=f"unknown goal kind {goal['kind']}")

            steps_taken += 1
            if success:
                goal["status"] = "done"
                consecutive_failures = 0
                self._log("goal_completed", kind=goal["kind"])
            else:
                goal["status"] = "failed"
                consecutive_failures += 1
                self._log("goal_failed", kind=goal["kind"], consecutive_failures=consecutive_failures)

            if consecutive_failures >= self.max_consecutive_failures:
                self.stopped_reason = "too_many_consecutive_failures"
                self._log(
                    "stopped", reason=self.stopped_reason,
                    consecutive_failures=consecutive_failures,
                )
                break

        if self.stopped_reason is None:
            self.stopped_reason = "all_goals_processed"
            self._log("finished", reason=self.stopped_reason)

        return self.summary()

    def summary(self):
        done = sum(1 for g in self.goal_queue if g["status"] == "done")
        failed = sum(1 for g in self.goal_queue if g["status"] == "failed")
        pending = sum(1 for g in self.goal_queue if g["status"] == "pending")
        return {
            "name": self.name,
            "total_goals": len(self.goal_queue),
            "done": done,
            "failed": failed,
            "pending": pending,
            "stopped_reason": self.stopped_reason,
            "trace_length": len(self.memory),
        }

    def explain_trace(self, last_n=None):
        entries = self.memory if last_n is None else self.memory[-last_n:]
        lines = []
        for entry in entries:
            detail = {k: v for k, v in entry.items() if k not in ("event", "timestamp")}
            lines.append(f"[{entry['event']}] {detail}")
        return "\n".join(lines)
