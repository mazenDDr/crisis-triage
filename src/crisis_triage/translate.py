"""Machine translation to English with NLLB-200, the step Laya needs before Haitian Creole.

T03 found Laya reads the human English translations of the Haiti SMS far better than the
Creole originals (+0.150 mean AUC on dev). This step replaces the human translator.
"""

from __future__ import annotations

# NLLB language codes for the tags data.guess_creole_or_french gives. `unk` is mostly Creole
# with no function word matched, so it is read as Creole.
NLLB_CODES = {"ht": "hat_Latn", "fr": "fra_Latn", "unk": "hat_Latn"}


class Translator:
    def __init__(self, model_id: str, device: str = "cuda", num_beams: int = 4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.float16)
        self.model.to(device).eval()
        self.model.generation_config.max_length = None  # max_new_tokens decides
        self.device, self.num_beams = device, num_beams
        self.eng = self.tok.convert_tokens_to_ids("eng_Latn")

    def __call__(
        self, texts: list[str], src: str, lowercase: bool = False, max_new_tokens: int = 128
    ) -> list[str]:
        """Translate texts that share one source language code (e.g. `hat_Latn`).

        `lowercase`: many SMS are written in capitals, which NLLB often leaves untranslated or
        turns into an unrelated sentence; T03b measures whether lower-casing first helps.
        """
        import torch

        if lowercase:
            texts = [t.lower() for t in texts]
        self.tok.src_lang = src
        batch = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            out = self.model.generate(
                **batch.to(self.device),
                forced_bos_token_id=self.eng,
                num_beams=self.num_beams,
                max_new_tokens=max_new_tokens,
            )
        return self.tok.batch_decode(out, skip_special_tokens=True)


# Chosen on the T03b dev sample (results/t03b_dev.json): the 1.3B model was no better
# (+0.004 mean AUC [-0.026, +0.033]) at 2x the time, lower-casing helped (+0.025 [-0.000, +0.048]),
# and greedy decoding was no faster one message at a time (p50 276 vs 269 ms, interleaved).
CHOSEN_MT = {"model": "facebook/nllb-200-distilled-600M", "lowercase": True, "num_beams": 4}
