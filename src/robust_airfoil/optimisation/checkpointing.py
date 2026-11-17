"""Hash-verified local NSGA-II checkpoints with frozen scientific context.

Only load checkpoints created by this authorised local campaign. A checksum
verifies continuity, not the safety of arbitrary externally supplied pickle.
The large model and geometry index are rebound from separately verified inputs.
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

from robust_airfoil.hashing import hash_object, sha256_file


def save_algorithm_checkpoint(algorithm: Any, directory: Path, context: dict[str, Any]) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    detached: list[tuple[Any,str,Any]] = []
    scientific_counters = None
    for obj,attribute in [(algorithm.problem,'candidate_evaluator'),(algorithm.callback,'evaluator')]:
        if hasattr(obj,attribute):
            value = getattr(obj,attribute)
            detached.append((obj,attribute,value))
            if value is not None and hasattr(value,'counters'):
                scientific_counters = value.counters.copy()
            setattr(obj,attribute,None)
    target = directory/'algorithm_checkpoint.pkl'
    receipt = directory/'algorithm_checkpoint.json'
    pending = directory/'algorithm_checkpoint.next.pkl'
    try:
        state = {'algorithm':algorithm,'context_hash':hash_object(context),
                 'numpy_random_state':np.random.get_state(),'python_random_state':random.getstate(),
                 'scientific_counters':scientific_counters}
        with pending.open('wb') as handle:
            pickle.dump(state,handle,protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        for obj,attribute,value in detached:
            setattr(obj,attribute,value)
    metadata = {'schema':'local-nsga2-checkpoint-v1','context':context,'context_hash':hash_object(context),
                'generation_completed':int(algorithm.n_gen)-1,'sha256':sha256_file(pending),'bytes':pending.stat().st_size}
    if target.is_file() and receipt.is_file():
        target.replace(directory/'algorithm_checkpoint.previous.pkl')
        receipt.replace(directory/'algorithm_checkpoint.previous.json')
    pending.replace(target)
    next_receipt = directory/'algorithm_checkpoint.next.json'
    next_receipt.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    next_receipt.replace(receipt)


def load_algorithm_checkpoint(directory: Path, context: dict[str, Any], *, problem: Any = None, evaluator: Any = None) -> Any:
    target = directory/'algorithm_checkpoint.pkl'
    metadata = json.loads((directory/'algorithm_checkpoint.json').read_text(encoding='utf-8'))
    if metadata.get('context_hash') != hash_object(context) or metadata.get('context') != context:
        raise ValueError('Checkpoint scientific context mismatch')
    if not target.is_file() or metadata.get('sha256') != sha256_file(target):
        raise ValueError('Checkpoint hash mismatch; refuse before deserialisation')
    with target.open('rb') as handle:
        state = pickle.load(handle)
    if state['context_hash'] != hash_object(context):
        raise ValueError('Checkpoint internal context mismatch')
    algorithm = state['algorithm']
    if problem is not None:
        algorithm.problem = problem
    if hasattr(algorithm.problem,'candidate_evaluator'):
        if evaluator is None:
            raise ValueError('The verified scientific evaluator must be supplied on resume')
        algorithm.problem.candidate_evaluator = evaluator
    if hasattr(algorithm.callback,'evaluator'):
        if evaluator is None:
            raise ValueError('The scientific callback requires its verified evaluator')
        algorithm.callback.evaluator = evaluator
    if evaluator is not None and state['scientific_counters'] is not None:
        evaluator.counters = state['scientific_counters']
    np.random.set_state(state['numpy_random_state'])
    random.setstate(state['python_random_state'])
    return algorithm


def assert_campaign_resources(directory: Path) -> dict[str, int]:
    """Protect workstation headroom without changing the scientific budget."""
    import shutil

    import psutil

    available = int(psutil.virtual_memory().available)
    free_disk = int(shutil.disk_usage(directory).free)
    if available < 3*1024**3:
        raise RuntimeError('Production memory floor reached; preserve and resume the checkpoint')
    if free_disk < 20*1024**3:
        raise RuntimeError('Production disk floor reached; preserve and resume the checkpoint')
    return {'available_memory_bytes':available,'free_disk_bytes':free_disk}
