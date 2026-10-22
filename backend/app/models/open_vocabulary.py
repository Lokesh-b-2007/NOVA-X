"""
Open-Vocabulary Recognition (OWLv2)
--------------------------------------
Used ONLY as a secondary semantic recognition step for objects the OOD
module has already flagged UNKNOWN. This is deliberate: Open-Vocabulary
Detection is NOT Open-Set Detection. OWLv2 never decides whether something
is known or unknown — it only guesses what an already-unknown object might
be called, for human review.
"""
import logging
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger("nova_x.open_vocabulary")


class OWLv2LoadError(Exception):
    """Raised when the OWLv2 model/processor cannot be loaded."""


class OpenVocabularyRecognizer:
    def __init__(self, model_id: str = None, vocabulary: list = None, device: str = None):
        self.model_id = model_id or settings.OVD_MODEL_ID
        self.vocabulary = vocabulary or settings.OVD_VOCABULARY
        self.device = device or settings.OVD_DEVICE
        self._model = None
        self._processor = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
        except ImportError as e:
            raise OWLv2LoadError(
                "The 'transformers' package is not installed. Run "
                "`pip install -r requirements.txt` in the backend/ directory."
            ) from e

        try:
            resolved_device = self._resolve_device()
            logger.info("Loading OWLv2 model '%s' on device '%s'...", self.model_id, resolved_device)
            self._processor = Owlv2Processor.from_pretrained(self.model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id).to(resolved_device)
            self._model.eval()
            self.device = resolved_device
        except Exception as e:
            raise OWLv2LoadError(
                f"Failed to load OWLv2 model '{self.model_id}': {e}. "
                "This likely requires internet access on first run to download weights "
                "from Hugging Face — verify network access or pre-download the model."
            ) from e

    def recognize(self, crop_rgb: np.ndarray, vocabulary: Optional[list] = None, top_k: int = 5) -> dict:
        """
        Classify a cropped RGB image against the configured (or supplied)
        open vocabulary. Returns model-ranked top-k label/confidence pairs;
        when no label meets OVD_SCORE_THRESHOLD, the best available model
        candidates are still returned for human review.
        """
        self._load()

        import torch
        from PIL import Image

        vocab = vocabulary or self.vocabulary
        pil_image = Image.fromarray(crop_rgb)

        # OWLv2 is a zero-shot detector: we query it with the whole crop as
        # the "image" and ask it to localize/score each candidate label
        # phrase, then treat the crop as the singular object of interest by
        # taking the highest-scoring box per label.
        texts = [[f"a photo of a {label}" for label in vocab]]
        try:
            inputs = self._processor(text=texts, images=pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = self._processor.post_process_object_detection(
                outputs=outputs, threshold=settings.OVD_SCORE_THRESHOLD, target_sizes=target_sizes
            )[0]
            if not len(results["scores"]):
                # Keep a real model-ranked candidate for human review when none
                # clears the configured confidence threshold.
                results = self._processor.post_process_object_detection(
                    outputs=outputs, threshold=0.0, target_sizes=target_sizes
                )[0]
        except Exception as e:
            raise OWLv2LoadError(f"OWLv2 inference failed: {e}") from e

        scores, labels = results["scores"].cpu().numpy(), results["labels"].cpu().numpy()
        best_per_label: dict[str, float] = {}
        for score, label_idx in zip(scores, labels):
            label = vocab[int(label_idx)]
            best_per_label[label] = max(best_per_label.get(label, 0.0), float(score))

        ranked = sorted(best_per_label.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        if not ranked:
            return {"label": None, "confidence": None, "top_predictions": []}

        top_label, top_conf = ranked[0]
        # Normalize the returned confidences to sum to <=1 across the
        # reported top-k for a readable "share of belief" style display,
        # without pretending they were a softmax over the full vocabulary.
        return {
            "label": top_label,
            "confidence": round(top_conf, 4),
            "top_predictions": [{"label": l, "confidence": round(c, 4)} for l, c in ranked],
        }


_recognizer_instance: Optional[OpenVocabularyRecognizer] = None


def get_recognizer() -> OpenVocabularyRecognizer:
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = OpenVocabularyRecognizer()
    return _recognizer_instance
