"""Same seed, same waveform — and how much faster.

THE ONLY ACCEPTANCE CRITERION IS BIT IDENTITY. A speedup that changes the
voice is not a speedup, it is a different model, and there is no way to argue
about "almost the same" once a reference recording is involved.

Which is why the seed is set immediately before each run and why nothing in
the grafts touches sampling. Both runs draw from the same stream, in the same
order, with the same shapes. If a graft ever reduced a logit tensor to the
range that actually gets used, the noise draw in `_sample` would change shape
and the voice would drift — correctly, and invisibly, which is the dangerous
kind.

    python bench/verify.py --reference ~/.them/voices/samantha2-fr
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

MODEL = "Edge0/Audio8-TTS-Realistic-0.6b"
SEED = 123

CASES = [
    ("fr", "J'allais dire quelque chose là-dessus, mais tu as parlé avant moi."),
    ("fr-long", "Je ne sais pas si c'est une bonne idée. On a essayé la semaine "
                "dernière et ça n'a rien donné, mais le contexte a changé depuis."),
    ("fr-court", "D'accord."),
]


def render(model, inputs, max_new_tokens=1024):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            return_dict_in_generate=True,
        )
        waveforms, lengths = model.decode_audio(out.codes)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    audio = waveforms[0, : int(lengths[0])].float().cpu().numpy()
    return audio, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(Path.home() / ".them/voices/samantha2-fr"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    folder = Path(args.reference).expanduser()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16)
        .eval()
        .to("cuda")
    )

    def prepared(text: str) -> dict:
        got = processor(
            text=[text],
            reference_audio=[str(folder / "reference.wav")],
            reference_text=[(folder / "reference.txt").read_text(encoding="utf-8").strip()],
            return_tensors="pt",
        )
        return {name: value.to("cuda") for name, value in got.items()}

    prompts = {name: prepared(text) for name, text in CASES}

    # Une chauffe jetée avant TOUTE mesure, et avant la greffe : la première
    # génération paie l'allocation et le choix d'algorithmes de cuBLAS, et la
    # compter ferait passer le chemin d'origine pour plus lent qu'il n'est.
    render(model, prompts[CASES[0][0]], max_new_tokens=64)

    before = {}
    for name, _ in CASES:
        audio, seconds = render(model, prompts[name])
        before[name] = (audio, seconds)
        print(f"origine  {name:9} {len(audio) / model.config.codec_sample_rate:5.2f} s "
              f"en {seconds:6.2f} s = {len(audio) / model.config.codec_sample_rate / seconds:.3f}x",
              flush=True)

    from faster_audio8 import accelerate

    grafts = accelerate(model)
    render(model, prompts[CASES[0][0]], max_new_tokens=64)  # chauffe la capture

    print()
    identical = 0
    for name, _ in CASES:
        audio, seconds = render(model, prompts[name])
        ref, ref_seconds = before[name]
        same = audio.shape == ref.shape and np.array_equal(audio, ref)
        identical += int(same)
        verdict = "identique au bit" if same else (
            f"DIFFERENT (ecart max {np.abs(audio[:min(len(audio), len(ref))] - ref[:min(len(audio), len(ref))]).max():.2e})"
            if audio.shape == ref.shape else
            f"DIFFERENT (longueurs {len(audio)} contre {len(ref)})")
        print(f"greffe   {name:9} {len(audio) / model.config.codec_sample_rate:5.2f} s "
              f"en {seconds:6.2f} s = {len(audio) / model.config.codec_sample_rate / seconds:.3f}x "
              f"| x{ref_seconds / seconds:.2f} | {verdict}", flush=True)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            sf.write(f"{args.out}/{name}-greffe.wav", audio, model.config.codec_sample_rate)
            sf.write(f"{args.out}/{name}-origine.wav", ref, model.config.codec_sample_rate)

    print()
    for name, graft in grafts.items():
        print(f"{name}: {getattr(graft, 'replays', 0)} rejoues, "
              f"{getattr(graft, 'fallbacks', 0)} replis, "
              f"{getattr(graft, 'failures', [])} echecs")
    print(f"\n{identical}/{len(CASES)} identiques au bit")


if __name__ == "__main__":
    main()
