"""Où va le temps d'Audio8, par composant, sur la carte.

La méthode est celle de faster-ming : ne rien déduire, compter. On greffe sur
les trois méthodes qui font tout le travail — le gros transformeur, le petit,
et la tête de sortie — et on relève le nombre d'appels et le temps cumulé.

`torch.cuda.synchronize` à chaque borne, sinon on mesure la vitesse à laquelle
le processeur remplit la file et pas celle à laquelle la carte la vide — et
c'est précisément l'erreur qui ferait conclure que l'ordonnancement n'est pas
le goulot alors qu'il l'est.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

MODELE = "Edge0/Audio8-TTS-Realistic-0.6b"
VOIX = Path.home() / ".them" / "voices"
PHRASE = "J'allais dire quelque chose là-dessus, mais tu as parlé avant moi."

compte: dict[str, int] = defaultdict(int)
duree: dict[str, float] = defaultdict(float)


def greffe(objet, nom: str, etiquette: str) -> None:
    origine = getattr(objet, nom)

    def enveloppe(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = origine(*a, **k)
        torch.cuda.synchronize()
        duree[etiquette] += time.perf_counter() - t
        compte[etiquette] += 1
        return r

    setattr(objet, nom, enveloppe)


def main() -> None:
    processeur = AutoProcessor.from_pretrained(MODELE, trust_remote_code=True)
    modele = (
        AutoModel.from_pretrained(MODELE, trust_remote_code=True, dtype=torch.bfloat16)
        .eval()
        .to("cuda")
    )

    dossier = VOIX / "samantha2-fr"
    entrees = processeur(
        text=[PHRASE],
        reference_audio=[str(dossier / "reference.wav")],
        reference_text=[(dossier / "reference.txt").read_text(encoding="utf-8").strip()],
        return_tensors="pt",
    )
    entrees = {n: v.to("cuda") for n, v in entrees.items()}

    def tourne():
        with torch.inference_mode():
            s = modele.generate(
                **entrees, max_new_tokens=1024, temperature=0.8, top_p=0.95,
                top_k=50, do_sample=True, return_dict_in_generate=True,
            )
            ondes, longueurs = modele.decode_audio(s.codes)
        return ondes, longueurs

    import sys
    if "--greffe" in sys.argv:
        from faster_audio8 import accelerate
        accelerate(modele)

    tourne()  # chauffe jetée

    compte.clear()
    duree.clear()
    greffe(modele, "_slow_step", "gros transformeur, 24 couches")
    greffe(modele, "_fast_step", "petit transformeur, 4 couches")
    greffe(modele, "_generate_codebooks", "une trame de codes")
    greffe(modele, "decode_audio", "codec")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ondes, longueurs = tourne()
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    secondes = int(longueurs[0]) / modele.config.codec_sample_rate
    print(f"\n{secondes:.2f} s de parole en {total:.2f} s  "
          f"= {secondes / total:.3f}x le temps reel\n")
    print(f"{'composant':34} {'appels':>7} {'cumul s':>9} {'par appel':>11} {'part':>7}")
    for nom in sorted(duree, key=lambda k: -duree[k]):
        ms = duree[nom] / compte[nom] * 1000
        print(f"{nom:34} {compte[nom]:7d} {duree[nom]:9.2f} {ms:9.2f} ms "
              f"{duree[nom] / total * 100:6.1f} %")
    reste = total - duree["gros transformeur, 24 couches"] - duree["une trame de codes"] \
        - duree["codec"]
    print(f"{'le reste (echantillonnage, boucle)':34} {'':7} {reste:9.2f} "
          f"{'':11} {reste / total * 100:6.1f} %")


if __name__ == "__main__":
    main()
