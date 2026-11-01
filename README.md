# faster-audio8

[Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b)
inference, about **4.6x to 5x faster**, waveform **identical to the bit**.

```python
from transformers import AutoModel
from faster_audio8 import accelerate

model = AutoModel.from_pretrained(
    "Edge0/Audio8-TTS-Realistic-0.6b", trust_remote_code=True, dtype=torch.bfloat16
).eval().to("cuda")

accelerate(model)          # in place, idempotent
```

Nothing else changes. Same `generate`, same processor, same sampling, same
seed, same voice.

## What it does

Measured on an **RTX 3070 Ti Laptop**, bfloat16, French with a reference voice:

| | before | after | |
|---|---:|---:|---:|
| 4.3 s utterance | 0.27x realtime | **1.25x** | x4.6 |
| 8.7 s utterance | 0.27x realtime | **1.35x** | x5.0 |
| 0.8 s utterance | 0.26x realtime | **0.77x** | x3.0 |

Below 1.0x a voice falls further behind with every clause and never catches up
inside a turn, so crossing it is the whole point. The short case stays under
because a one-second utterance is mostly prefill, which is untouched.

**3/3 bit identical**, by `numpy.array_equal` on the decoded waveform.

## Why it works

The model is not compute bound. Profiled before any change:

| component | calls | per call | share |
|---|---:|---:|---:|
| fast transformer, 4 layers | 1030 | 9.66 ms | 55 % |
| slow transformer, 24 layers | 103 | 60.60 ms | 35 % |
| codec | 1 | 143 ms | 1 % |

Nine milliseconds for four layers of width 896, and sixty for twenty-four, is
not arithmetic. The **kernel launch floor on this machine is 17.7 us** — WDDM,
the same figure its sibling project faster-ming measured — which puts those
numbers at five hundred and three thousand launches per step respectively, for
work that is tens of microseconds to two milliseconds wide.

So both decode steps are captured into CUDA graphs. Nothing touches the
arithmetic; every change removes launches.

| component | before | after |
|---|---:|---:|
| fast transformer | 9.66 ms | **1.03 ms** |
| slow transformer | 60.60 ms | **14.01 ms** |

## The two traps, and they are the same trap

**A CUDA graph records addresses, not values.** Everything it reads must stay
alive and stay put. Both failures here were versions of that, and neither one
says what it is:

1. **`generate()` rebuilds the KV caches on every call.** The old buffers go
   back to the allocator, the next allocation writes over them, and a graph
   captured during the previous utterance reads and writes somebody else's
   memory. The symptom is a device-side assert deep inside an RMS norm. Both
   grafts therefore remember the cache by pointer and rebuild themselves when
   the model swaps it.

2. **The capture itself writes into the cache.** Three warmups plus the
   capture are four passes that lay keys and values at the position they were
   built for. The fast transformer does not care — its cache is overwritten
   every frame. The slow one carries the whole prompt, and the capture was
   quietly corrupting its first token. The symptom was an utterance that was
   *plausible and one frame short*: a displaced logit moves a draw, and a moved
   draw moves the instant she decides she has finished. The slow graft saves
   and restores that slice around the capture.

## What was measured and rejected

**Restricting the output head.** At every slow step the model projects to all
155,776 vocabulary columns, and one line later a filter sets all but 4,097 of
them to minus infinity — **97.4 % of that matmul is thrown away**. Restricting
it is a 58x saving on the CPU, where the head is 27 % of a step. On the card
it is 0.785 ms against 0.039, inside a step that costs 60: **one and a half per
cent**, not worth it. And it would change the shape handed to `torch.rand`
inside `_sample`, which moves the random stream and therefore the voice.

Not re-proposing this is the point of writing it down.

## What is left

At 1.25x the remaining time is roughly: half in the per-frame code generation,
a third in the slow step, and an eighth in sampling and loop overhead.

- **The slow step still costs 14 ms.** The KV cache is built with
  `return_full=True`, so attention reads all 2,048 cached positions on every
  step regardless of how many are real. Trimming it would cut real arithmetic,
  not launches — a different kind of change, and one that has to be re-checked
  for bit identity rather than assumed.
- **Sampling costs around 10 ms per frame.** `_sample_semantic` runs the
  processors twice over the full vocabulary, and each of the ten codebooks
  runs them once more. Capturable in principle, delicate in practice, because
  this is exactly where the random draws live.

## Requirements

A CUDA card, PyTorch with graph support, and `transformers>=4.57`. Torch is
deliberately not pinned: the right wheel depends on the CUDA version you have.

```
python bench/verify.py --reference ~/.them/voices/samantha2-fr
python bench/profile_step.py --greffe
```

MIT. The model itself is Apache 2.0 and belongs to Edge0.
