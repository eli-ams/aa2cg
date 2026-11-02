"""CPU availability without importing the scientific backend."""
import os


def available_cores():
    """Return available logical CPUs, respecting process affinity where supported."""
    counts = []
    for name, args in (("process_cpu_count", ()), ("sched_getaffinity", (0,)),
                       ("cpu_count", ())):
        query = getattr(os, name, None)
        if query is None:
            continue
        try:
            value = query(*args)
            count = len(value) if name == "sched_getaffinity" else value
            if count and count > 0:
                counts.append(int(count))
        except (OSError, NotImplementedError):
            continue
    return max(1, min(counts)) if counts else 1
