import time
import uuid
from core.similarity import cosine_similarity


class Episode:
    def __init__(self, content, context_vector, tags=None, timestamp=None):
        self.episode_id = str(uuid.uuid4())
        self.content = content
        self.context_vector = context_vector
        self.tags = tags or []
        self.timestamp = timestamp or time.time()
        self.recall_count = 0
        self.importance = 1.0

    def touch(self):
        self.recall_count += 1
        self.importance = min(5.0, self.importance + 0.1)


class EpisodicMemory:
    def __init__(self, max_episodes=5000):
        self.episodes = []
        self.max_episodes = max_episodes
        self.tag_index = {}

    def store(self, content, context_vector, tags=None):
        episode = Episode(content, context_vector, tags)
        self.episodes.append(episode)
        for tag in episode.tags:
            self.tag_index.setdefault(tag, []).append(episode)
        if len(self.episodes) > self.max_episodes:
            self._forget_least_important()
        return episode

    def _forget_least_important(self):
        self.episodes.sort(key=lambda e: (e.importance, e.timestamp))
        removed = self.episodes.pop(0)
        for tag in removed.tags:
            if removed in self.tag_index.get(tag, []):
                self.tag_index[tag].remove(removed)

    def retrieve_similar(self, query_vector, top_k=5, min_score=0.0):
        scored = []
        for episode in self.episodes:
            score = cosine_similarity(query_vector, episode.context_vector)
            recency_bonus = self._recency_bonus(episode)
            total_score = score * 0.85 + recency_bonus * 0.15
            if total_score >= min_score:
                scored.append((total_score, episode))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = scored[:top_k]
        for _, episode in results:
            episode.touch()
        return [(score, episode.content) for score, episode in results]

    def _recency_bonus(self, episode):
        age = time.time() - episode.timestamp
        return max(0.0, 1.0 - age / (86400 * 30))

    def retrieve_by_tag(self, tag, top_k=10):
        candidates = self.tag_index.get(tag, [])
        candidates = sorted(candidates, key=lambda e: e.timestamp, reverse=True)
        return [e.content for e in candidates[:top_k]]

    def summarize_recent(self, n=10):
        recent = sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)[:n]
        return [e.content for e in recent]

    def consolidate(self, similarity_threshold=0.92):
        merged = []
        used = set()
        for i, episode_a in enumerate(self.episodes):
            if i in used:
                continue
            cluster = [episode_a]
            for j in range(i + 1, len(self.episodes)):
                if j in used:
                    continue
                episode_b = self.episodes[j]
                score = cosine_similarity(episode_a.context_vector, episode_b.context_vector)
                if score >= similarity_threshold:
                    cluster.append(episode_b)
                    used.add(j)
            if len(cluster) > 1:
                merged_content = {
                    "merged_from": [e.episode_id for e in cluster],
                    "representative": cluster[0].content,
                    "count": len(cluster),
                }
                merged.append(merged_content)
        return merged

    def size(self):
        return len(self.episodes)

