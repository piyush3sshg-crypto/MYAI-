import time
import uuid
from Datastructures import UpdatablePriorityQueue


class ReasoningTask:
    def __init__(self, name, callback, priority=1.0, metadata=None):
        self.task_id = str(uuid.uuid4())
        self.name = name
        self.callback = callback
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.completed = False
        self.result = None
        self.error = None

    def execute(self):
        try:
            self.result = self.callback()
        except Exception as exc:
            self.error = exc
        self.completed = True
        return self.result


class TaskScheduler:
    def __init__(self):
        self.queue = UpdatablePriorityQueue()
        self.tasks = {}
        self.history = []
        self.max_history = 500

    def submit(self, name, callback, priority=1.0, metadata=None):
        task = ReasoningTask(name, callback, priority, metadata)
        self.tasks[task.task_id] = task
        self.queue.push(task.task_id, -priority)
        return task.task_id

    def run_next(self):
        if self.queue.is_empty():
            return None
        task_id, _ = self.queue.pop()
        task = self.tasks.pop(task_id)
        task.execute()
        self.history.append(task)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return task

    def run_all(self, max_tasks=1000):
        results = []
        count = 0
        while not self.queue.is_empty() and count < max_tasks:
            task = self.run_next()
            if task is not None:
                results.append(task)
            count += 1
        return results

    def pending_count(self):
        return len(self.queue)

    def cancel(self, task_id):
        if task_id in self.tasks:
            self.queue.remove(task_id)
            del self.tasks[task_id]
            return True
        return False

    def reprioritize(self, task_id, new_priority):
        if task_id in self.tasks:
            self.tasks[task_id].priority = new_priority
            self.queue.push(task_id, -new_priority)


class PeriodicJobRunner:
    def __init__(self):
        self.jobs = []

    def register(self, name, callback, interval_ticks):
        self.jobs.append({
            "name": name,
            "callback": callback,
            "interval": interval_ticks,
            "counter": 0,
        })

    def tick(self):
        fired = []
        for job in self.jobs:
            job["counter"] += 1
            if job["counter"] >= job["interval"]:
                job["counter"] = 0
                job["callback"]()
                fired.append(job["name"])
        return fired

