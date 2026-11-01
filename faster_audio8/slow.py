"""The slow transformer's decode step, in one CUDA graph.

Second by weight and first by cost per call: 103 calls at 60.6 ms on an RTX
3070 Ti Laptop, for twenty-four layers of width 896. At a 17.7 us launch
floor that is on the order of three thousand launches for a step whose real
arithmetic is around two milliseconds. The head over the full 155,776-token
vocabulary, which looks like the obvious culprit and is the one I expected,
measures 0.785 ms — one and a half per cent. It is launches, all of it.

THE ONE SHAPE THAT IS NOT CONSTANT, AND WHY IT DOES NOT MATTER.

Everything in the decode step has a fixed shape except the attention mask,
which the generation loop grows by one column per step. But `_causal_mask`
pads it to `max_seq_len` before using it, so what the layers actually see is
always the same width. We keep that padded buffer ourselves and write the
live mask into its head on each replay: identical arithmetic, constant shape.

The other two varying inputs are a physical cache position and a logical token
position. In the original both arrive as freshly built CUDA tensors — the
first via `torch.tensor([...], device=...)`, which is a host-to-device copy and
therefore the one thing a capture cannot contain. Both move into static
buffers.

WHAT STAYS OUTSIDE. The prefill — the first call, over the whole prompt — has
a different shape and runs once per utterance, so it falls back to the
original path and its keys and values land in the same cache the graph then
reads. And the sampling stays where it was: this graft returns logits over the
whole vocabulary, exactly as before, so the noise draw in `_sample` keeps its
shape and the voice its seed.

WHY NOT RESTRICT THE HEAD. 97.4 % of that 155,776-wide projection is thrown
away one line later by the semantic filter, and restricting it to the 4,097
columns that survive is a 58x saving on the matmul — on the CPU. On the card
it is 0.785 ms against 0.039, in a step that costs 60. It would also change
the shape fed to `torch.rand` inside `_sample`, which moves the random stream
and therefore the voice. Measured, rejected, written down so it is not
re-proposed: see the repository README.
"""

from __future__ import annotations

import torch

from .capture import Captured, capture


class GraphedSlowStep:
    """`_slow_step` as a replayable graph, for the decode shape only."""

    def __init__(self, model) -> None:
        self.model = model
        self.call = model._slow_step
        self.captured: Captured | bool | None = None
        self.failures: list[str] = []
        self.replays = 0
        self.fallbacks = 0
        self.rebuilds = 0
        self.cache_ptr = None
        config = model.config
        self.width = int(config.max_seq_len)
        self.rows = int(config.num_codebooks) + 1
        self.device = next(model.layers.parameters()).device

    def _cache_ptr(self):
        cache = getattr(self.model.layers[0].attention, "kv_cache", None)
        return None if cache is None else (cache.keys.data_ptr(), cache.values.data_ptr())

    def _build(self) -> Captured:
        model = self.model
        input_ids = torch.zeros((1, self.rows, 1), device=self.device, dtype=torch.long)
        cache_position = torch.zeros(1, device=self.device, dtype=torch.long)
        position_ids = torch.zeros((1, 1), device=self.device, dtype=torch.long)
        # Pleine largeur dès le départ, et à zéro : `_causal_mask` aurait padé
        # avec des zéros de toute façon, donc écrire la tête du tampon à
        # chaque pas rend exactement le même masque.
        attention_mask = torch.zeros(
            (1, self.width), device=self.device, dtype=torch.long
        )

        def corps():
            return model.__class__._slow_step(
                model, input_ids, cache_position, position_ids, attention_mask
            )

        # LA CAPTURE ÉCRIT DANS LE CACHE, ET LE CACHE N'EST PAS À ELLE.
        #
        # Trois chauffes plus la capture, c'est quatre passages qui posent des
        # clés et des valeurs à la position choisie — ici zéro, c'est-à-dire
        # le premier jeton du prompt, que le préremplissage vient d'écrire.
        # Pour le petit transformeur c'était sans effet : son cache est
        # réécrit en entier à chaque trame. Ici le cache PORTE la phrase.
        #
        # Le symptôme n'a rien dit de tout ça : l'onde restait plausible et
        # faisait une trame de moins, parce qu'un logit déplacé change un
        # tirage, et un tirage change l'instant où elle décide d'avoir fini.
        # Même famille que le gotcha de faster-ming, et il faut s'en méfier
        # pour TOUTE capture posée au milieu d'une génération.
        sauvegarde = [
            (couche.attention.kv_cache.keys[:, :, :1].clone(),
             couche.attention.kv_cache.values[:, :, :1].clone())
            for couche in model.layers
        ]
        try:
            graph, output, context = capture(corps)
        finally:
            for couche, (cles, valeurs) in zip(model.layers, sauvegarde):
                couche.attention.kv_cache.keys[:, :, :1].copy_(cles)
                couche.attention.kv_cache.values[:, :, :1].copy_(valeurs)
        return Captured(
            graph=graph,
            output=output,
            inputs={
                "input_ids": input_ids,
                "cache_position": cache_position,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            },
            held={"context": context},
        )

    def __call__(self, input_ids, cache_position, position_ids, attention_mask):
        decode = (
            tuple(input_ids.shape) == (1, self.rows, 1)
            and attention_mask.shape[0] == 1
            and attention_mask.shape[1] <= self.width
        )
        if not decode:
            self.fallbacks += 1
            return self.call(input_ids, cache_position, position_ids, attention_mask)

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
            return self.call(input_ids, cache_position, position_ids, attention_mask)

        tampon = self.captured.inputs["attention_mask"]
        # La queue reste à zéro d'un pas à l'autre : le masque ne fait que
        # croître, donc on n'efface jamais un un qui aurait dû rester.
        tampon[:, : attention_mask.shape[1]].copy_(attention_mask)
        self.captured.inputs["input_ids"].copy_(input_ids)
        self.captured.inputs["cache_position"].copy_(cache_position.reshape(1))
        self.captured.inputs["position_ids"].copy_(position_ids.reshape(1, 1))
        self.captured.graph.replay()
        self.replays += 1
        logits, hidden = self.captured.output
        return logits.clone(), hidden.clone()


def graft_slow(model) -> GraphedSlowStep:
    """Replace `model._slow_step` with its graphed twin. Idempotent."""
    existing = getattr(model, "_faster_audio8_slow", None)
    if existing is not None:
        return existing
    graphed = GraphedSlowStep(model)
    model._slow_step = graphed
    model._faster_audio8_slow = graphed
    return graphed


__all__ = ["GraphedSlowStep", "graft_slow"]
