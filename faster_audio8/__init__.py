"""Audio8-TTS inference, faster, waveform identical to the bit.

The model is not compute bound. Profiled on an RTX 3070 Ti Laptop, one French
utterance of 4.74 s in bfloat16, it spends 55 % of its time in a four-layer
transformer called 1030 times at 9.66 ms a call, and 35 % in a twenty-four
layer one called 103 times at 60.60 ms a call. The kernel launch floor on this
machine is 17.7 microseconds, which puts both numbers at hundreds to thousands
of launches per step for work that is tens of microseconds wide.

So nothing here touches the arithmetic. Every change removes launches.

    from faster_audio8 import accelerate
    accelerate(model)

`accelerate` is idempotent, and every graft falls back to the original call on
any shape it was not built for.
"""

from __future__ import annotations

from .capture import Captured, capture
from .fast import GraphedFastStep, graft_fast
from .slow import GraphedSlowStep, graft_slow

__all__ = ["accelerate", "Captured", "capture",
           "GraphedFastStep", "graft_fast",
           "GraphedSlowStep", "graft_slow"]


def accelerate(model, *, fast: bool = True, slow: bool = True) -> dict:
    """Graft what is enabled onto a loaded Audio8 model, in place.

    Returns the grafts, so a bench can read their replay and fallback counts.
    A graft that reports zero replays did not take, and that has to be visible
    rather than guessed at from the timing.
    """
    grafts: dict[str, object] = {}
    if fast:
        grafts["fast"] = graft_fast(model)
    if slow:
        grafts["slow"] = graft_slow(model)
    return grafts
