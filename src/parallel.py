"""Fork-based parallel map over index ranges.

Workers inherit large read-only inputs through fork (copy-on-write), so
nothing big is pickled on the way in; only each task's result comes back.
Where fork is unavailable, or n_jobs is 1, tasks run in-process.
"""

import multiprocessing as mp
import os

SHARED = {}


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def resolve_jobs(n_jobs):
    return cpu_count() if not n_jobs or n_jobs < 1 else n_jobs


def ranges(n, size):
    """Split range(n) into consecutive (start, end) tasks of at most `size` items."""
    size = max(1, int(size))
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def fork_map(func, tasks, n_jobs, **shared):
    """Return [func(task) for task in tasks], run on n_jobs forked workers.

    func must be a module-level function; it reads its inputs from SHARED.
    """
    SHARED.clear()
    SHARED.update(shared)
    try:
        if n_jobs <= 1 or len(tasks) <= 1 or "fork" not in mp.get_all_start_methods():
            return [func(t) for t in tasks]
        with mp.get_context("fork").Pool(min(n_jobs, len(tasks))) as pool:
            return pool.map(func, tasks, chunksize=1)
    finally:
        SHARED.clear()
