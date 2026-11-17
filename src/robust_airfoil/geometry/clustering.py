from __future__ import annotations

import numpy as np


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def cluster_coordinate_vectors(vectors: np.ndarray, rms_threshold: float = 1e-4) -> np.ndarray:
    union = UnionFind(len(vectors))
    for i in range(len(vectors)):
        distances = np.sqrt(np.mean((vectors[i + 1 :] - vectors[i]) ** 2, axis=1))
        for offset in np.flatnonzero(distances <= rms_threshold):
            union.union(i, i + 1 + int(offset))
    roots = [union.find(i) for i in range(len(vectors))]
    remap = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[root] for root in roots], dtype=int)
