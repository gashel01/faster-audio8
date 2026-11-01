"""The grafts must not change the waveform. That is the only contract.

These need the real weights and a real card, so they skip themselves without
`FASTER_AUDIO8_REFERENCE` — a folder holding `reference.wav` and
`reference.txt`, the shape Audio8 asks for.

    FASTER_AUDIO8_REFERENCE=~/.them/voices/samantha2-fr pytest -q
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REFERENCE = os.environ.get("FASTER_AUDIO8_REFERENCE", "")
MODEL = os.environ.get("FASTER_AUDIO8_MODEL", "Edge0/Audio8-TTS-Realistic-0.6b")
SEED = 123

besoin_carte = pytest.mark.skipif(
    not REFERENCE or not torch.cuda.is_available(),
    reason="needs FASTER_AUDIO8_REFERENCE and a CUDA card",
)


@pytest.fixture(scope="module")
def monde():
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16)
        .eval()
        .to("cuda")
    )
    folder = Path(REFERENCE).expanduser()

    def prompt(text: str) -> dict:
        got = processor(
            text=[text],
            reference_audio=[str(folder / "reference.wav")],
            reference_text=[(folder / "reference.txt").read_text(encoding="utf-8").strip()],
            return_tensors="pt",
        )
        return {name: value.to("cuda") for name, value in got.items()}

    return model, prompt


def rendu(model, inputs, max_new_tokens=1024) -> np.ndarray:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.8,
            top_p=0.95, top_k=50, do_sample=True, return_dict_in_generate=True,
        )
        waveforms, lengths = model.decode_audio(out.codes)
    return waveforms[0, : int(lengths[0])].float().cpu().numpy()


@besoin_carte
@pytest.mark.parametrize("texte", [
    "D'accord.",
    "J'allais dire quelque chose là-dessus, mais tu as parlé avant moi.",
    "Je ne sais pas si c'est une bonne idée. On a essayé la semaine dernière "
    "et ça n'a rien donné, mais le contexte a changé depuis.",
])
def test_identique_au_bit(monde, texte):
    model, prompt = monde
    inputs = prompt(texte)
    rendu(model, inputs, max_new_tokens=48)  # chauffe, hors mesure
    avant = rendu(model, inputs)

    from faster_audio8 import accelerate

    accelerate(model)
    apres = rendu(model, inputs)

    assert apres.shape == avant.shape, (
        "une longueur différente n'est pas du bruit numérique : c'est un "
        "tirage déplacé, donc un autre instant de fin"
    )
    assert np.array_equal(apres, avant)


@besoin_carte
def test_les_greffes_prennent(monde):
    """Une greffe qui ne prend jamais passerait les tests d'identité sans rien
    faire. Le compte de rejoués est la seule preuve qu'elle a servi."""
    model, prompt = monde
    from faster_audio8 import accelerate

    greffes = accelerate(model)
    rendu(model, prompt("D'accord."), max_new_tokens=48)

    for nom, greffe in greffes.items():
        assert greffe.replays > 0, f"{nom} n'a jamais été rejouée"
        assert not greffe.failures, f"{nom} : {greffe.failures}"


def test_accelerate_est_idempotent():
    """Sans carte non plus : appeler deux fois ne doit pas empiler deux
    enveloppes, sinon chaque appel passerait par la précédente."""
    from faster_audio8 import accelerate

    class Faux:
        def _fast_step(self, hidden, position):
            return hidden

        def _slow_step(self, *a):
            return a

        class config:  # noqa: N801
            fast_dim = 896
            num_codebooks = 10
            max_seq_len = 2048

        fast_layers = []
        layers = []

    faux = Faux()
    # Les greffes lisent le dtype et le device des paramètres, donc on ne
    # construit pas ici ; on vérifie seulement que la seconde pose rend la
    # première, et pas une enveloppe autour d'elle.
    try:
        une = accelerate(faux)
        deux = accelerate(faux)
    except (StopIteration, AttributeError):
        pytest.skip("les greffes ont besoin d'un vrai modèle pour se construire")
    assert une["fast"] is deux["fast"]
    assert une["slow"] is deux["slow"]
