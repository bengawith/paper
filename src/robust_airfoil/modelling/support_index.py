"""Exact nearest-state support queries exploiting repeated geometry per polar.

The geometry-tree radius is a lower bound on every unseen state's full
geometry-plus-incidence distance. Search expands until that bound proves the
returned neighbours exact. Missing incidence coverage is never approximated.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class GroupedSupportIndex:
    def __init__(self, features: np.ndarray):
        self.features = np.asarray(features,dtype=float)
        if self.features.ndim != 2 or not len(self.features) or self.features.shape[1] < 2:
            raise ValueError('Support index requires a nonempty feature matrix')
        if not np.isfinite(self.features).all():
            raise ValueError('Support features must be finite')
        self.n = len(self.features)
        self.geometries, inverse = np.unique(self.features[:,:-1],axis=0,return_inverse=True)
        self.tree = cKDTree(self.geometries)
        groups = [np.flatnonzero(inverse == i) for i in range(len(self.geometries))]
        width = max(max(map(len,groups)),5)
        self.angles = np.full((len(groups),width),np.inf)
        self.indices = np.full((len(groups),width),self.n,dtype=np.int64)
        for i,ids in enumerate(groups):
            self.angles[i,:len(ids)] = self.features[ids,-1]
            self.indices[i,:len(ids)] = ids
        self._angle_cache: dict[tuple[float,int],tuple[np.ndarray,np.ndarray]] = {}

    def _angular_neighbours(self, alpha: float, k: int) -> tuple[np.ndarray,np.ndarray]:
        key = (alpha,k)
        if key not in self._angle_cache:
            costs = (self.angles-alpha)**2
            indices = self.indices
            if costs.shape[1] < k:
                padding = k-costs.shape[1]
                costs = np.pad(costs,((0,0),(0,padding)),constant_values=np.inf)
                indices = np.pad(indices,((0,0),(0,padding)),constant_values=self.n)
            chosen = np.argpartition(costs,k-1,axis=1)[:,:k]
            result = np.take_along_axis(costs,chosen,axis=1),np.take_along_axis(indices,chosen,axis=1)
            if len(self._angle_cache) >= 256:
                self._angle_cache.pop(next(iter(self._angle_cache)))
            self._angle_cache[key] = result
        return self._angle_cache[key]

    def _query_chunk(self, points: np.ndarray, k: int) -> tuple[np.ndarray,np.ndarray]:
        output = np.empty((len(points),k),float)
        output_ids = np.empty((len(points),k),np.int64)
        pending = np.arange(len(points))
        width = min(8,len(self.geometries))
        while len(pending):
            current = points[pending]
            unique, inverse = np.unique(current[:,:-1],axis=0,return_inverse=True)
            distances, neighbours = self.tree.query(unique,k=width)
            distances = np.asarray(distances).reshape(len(unique),width)[inverse]
            neighbours = np.asarray(neighbours).reshape(len(unique),width)[inverse]
            values = np.empty((len(current),width,k),float)
            row_ids = np.empty((len(current),width,k),np.int64)
            for alpha in np.unique(current[:,-1]):
                selected = np.flatnonzero(current[:,-1] == alpha)
                angular, ids = self._angular_neighbours(float(alpha),k)
                values[selected] = distances[selected,:,None]**2 + angular[neighbours[selected]]
                row_ids[selected] = ids[neighbours[selected]]
            flat_values = values.reshape(len(current),-1)
            flat_row_ids = row_ids.reshape(len(current),-1)
            chosen = np.argpartition(flat_values,k-1,axis=1)[:,:k]
            best = np.take_along_axis(flat_values,chosen,axis=1)
            ids = np.take_along_axis(flat_row_ids,chosen,axis=1)
            order = np.argsort(best,axis=1,kind='stable')
            best = np.take_along_axis(best,order,axis=1)
            ids = np.take_along_axis(ids,order,axis=1)
            complete = np.ones(len(current),bool) if width == len(self.geometries) else best[:,-1] <= distances[:,-1]**2
            output[pending[complete]] = np.sqrt(best[complete])
            output_ids[pending[complete]] = ids[complete]
            pending = pending[~complete]
            width = min(width*2,len(self.geometries))
        return output,output_ids

    def query(self, points: np.ndarray, k: int = 5) -> tuple[np.ndarray,np.ndarray]:
        values = np.asarray(points,dtype=float)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        if values.ndim != 2 or values.shape[1] != self.features.shape[1] or not np.isfinite(values).all():
            raise ValueError('Query shape or finite-value contract failed')
        if not isinstance(k,int) or k < 1:
            raise ValueError('Neighbour count must be a positive integer')
        if k > 32:
            return cKDTree(self.features).query(points,k=k)
        distances, indices = np.empty((len(values),k)),np.empty((len(values),k),np.int64)
        for start in range(0,len(values),4096):
            distances[start:start+4096],indices[start:start+4096] = self._query_chunk(values[start:start+4096],k)
        if k == 1:
            distances,indices = distances[:,0],indices[:,0]
        if single:
            distances,indices = distances[0],indices[0]
        return distances,indices
