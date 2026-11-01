"""The fast transformer, in one CUDA graph.

WHY THIS ONE FIRST. Profiled on an RTX 3070 Ti Laptop, one French utterance of
4.74 s, in bfloat16:

    fast transformer, 4 layers    1030 calls    9.95 s    9.66 ms each   55 %
    slow transformer, 24 layers    103 calls    6.24 s   60.60 ms each   35 %
    codec                            1 call     0.14 s                    1 %

Nine and a half milliseconds for four layers of width 896 is not arithmetic.
The kernel launch floor measured on this machine is **17.7 us** -- WDDM, same
as the number faster-ming measured -- so 9.66 ms is on the order of five
hundred launches for a step whose real work is tens of microseconds. The model
is not compute bound anywhere. It is launch bound everywhere, and this is the
hottest loop: ten calls per audio frame, 21.5 frames per second of speech.

WHAT MAKES IT CAPTURABLE, AND IT IS ALMOST A TEXTBOOK CASE.

Every shape in this step is constant. The hidden state is one frame wide, the
key/value cache is exactly `num_codebooks` long and is overwritten each frame,
and the causal mask is built against that same fixed length. The only thing
that varies between the ten calls is the position, 0 through 9 -- and in the
original that position arrives as `torch.tensor([position], device=...)`, which
is a host-to-device copy and therefore the one thing a capture cannot contain.

So the position moves into a static buffer that the caller writes before each
replay. Everything else -- the rotary lookup, the mask, the four layers, the
output projection -- is device work with fixed shapes, and goes inside.

WHAT IS DELIBERATELY LEFT OUTSIDE. Nothing is sampled here. `_fast_step`
returns scores and the caller draws from them, so the random stream is
untouched by this graft -- which is what lets the waveform stay identical to
the bit under a fixed seed. See `verify.py`.

THE TRAP THIS ONE SPRANG, AND IT IS THE FAMILY TRAP IN A NEW DRESS.

A graph records addresses. faster-ming learnt that about a *constant* left in
a local. Here it is worse and quieter: `generate()` calls
`_setup_generation_caches`, which builds **new** `ArkttsKVCache` modules on
every single call. The old buffers go back to the allocator, the next large
allocation writes over them, and a graph captured during the previous
utterance then reads and writes somebody else's memory. The symptom is a
device-side assert several layers down, in an RMS norm, which says nothing
about caches at all.

So the graft remembers which cache it was captured against, by pointer, and
rebuilds itself when the model swaps them. Cheap -- one comparison per call,
one rebuild per utterance -- and it is the only honest answer: there is no
way to hold a buffer the model itself intends to replace.
"""

from __future__ import annotations

import torch

from .capture import Captured, capture


class GraphedFastStep:
    """`_fast_step` as a replayable graph, with the position as an input.

    Falls back to the original call for any shape it was not built for. That
    is not defensive decoration: the first call of each frame passes the
    projected slow hidden state, and a batch larger than one, or a model
    reconfigured between runs, must still produce audio rather than an error.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.call = model._fast_step
        self.captured: Captured | bool | None = None
        self.failures: list[str] = []
        self.replays = 0
        self.fallbacks = 0
        self.rebuilds = 0
        self.cache_ptr = None
        config = model.config
        self.shape = (1, 1, int(config.fast_dim))
        self.dtype = next(model.fast_layers.parameters()).dtype
        self.device = next(model.fast_layers.parameters()).device

    def _build(self) -> Captured:
        model = self.model
        config = model.config
        # Les trois tampons que le graphe LIT, gardés par `Captured` : un
        # graphe enregistre des ADRESSES, et tout ce qu'il lit doit survivre
        # aussi longtemps que lui. Y compris le masque de clés, qui est une
        # constante — c'est exactement l'oubli qui a coûté une heure à
        # faster-ming, et il ne se voit qu'à retardement.
        hidden = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        position = torch.zeros(1, device=self.device, dtype=torch.long)
        key_mask = torch.ones(
            (1, int(config.num_codebooks)), device=self.device, dtype=torch.bool
        )

        def corps():
            rope = model.fast_freqs_cis[position]
            mask = model._causal_mask(key_mask, position, int(config.num_codebooks))
            h = hidden
            for layer in model.fast_layers:
                h = layer(h, rope, mask, position)
            return model.fast_output(model.fast_norm(h))[:, -1]

        graph, output, context = capture(corps)
        return Captured(
            graph=graph,
            output=output,
            inputs={"hidden": hidden, "position": position},
            held={"context": context, "key_mask": key_mask},
        )

    def _cache_ptr(self):
        """L'identité des tampons que le graphe écrit, telle que le modèle la voit.

        Par POINTEUR et pas par `is` : deux objets `ArkttsKVCache` différents
        peuvent recevoir la même mémoire de l'allocateur, et c'est précisément
        ce cas-là qui est dangereux — le graphe continuerait d'écrire au bon
        endroit par accident, jusqu'à ce qu'il n'y soit plus.
        """
        cache = getattr(self.model.fast_layers[0].attention, "kv_cache", None)
        return None if cache is None else (cache.keys.data_ptr(), cache.values.data_ptr())

    def __call__(self, hidden: torch.Tensor, position: int):
        if tuple(hidden.shape) != self.shape or hidden.dtype != self.dtype:
            self.fallbacks += 1
            return self.call(hidden, position)
        # Le modèle a-t-il remplacé ses caches depuis la capture ? Voir le
        # haut du fichier : c'est la panne la plus trompeuse du lot.
        pointeur = self._cache_ptr()
        if self.captured and pointeur != self.cache_ptr:
            self.captured = None
            self.rebuilds += 1
        if self.captured is None:
            try:
                self.captured = self._build()
                self.cache_ptr = pointeur
            except Exception as exc:  # pragma: no cover - driver dependent
                self.failures.append(repr(exc))
                self.captured = False
        if self.captured is False:
            self.fallbacks += 1
            return self.call(hidden, position)
        self.replays += 1
        return self.captured.replay(
            hidden=hidden,
            position=torch.tensor(
                [position], device=self.device, dtype=torch.long
            ),
        )


def graft_fast(model) -> GraphedFastStep:
    """Replace `model._fast_step` with its graphed twin. Idempotent."""
    existing = getattr(model, "_faster_audio8_fast", None)
    if existing is not None:
        return existing
    graphed = GraphedFastStep(model)
    model._fast_step = graphed
    model._faster_audio8_fast = graphed
    return graphed


__all__ = ["GraphedFastStep", "graft_fast"]
