import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
from faster_whisper import WhisperModel
from app.config import settings

logger = logging.getLogger("whisper.engine")


class WhisperEngine:
    def __init__(self):
        self._models: Dict[str, WhisperModel] = {}
        self._is_loaded = False

    @property
    def loaded_models(self) -> List[str]:
        return list(self._models.keys())

    def load_models(self):
        """Loads all models specified in WHISPER_MODELS into memory."""
        os.makedirs(settings.whisper_models_dir, exist_ok=True)
        for model_name in settings.models_list:
            if model_name in self._models:
                continue

            model_dir = os.path.join(settings.whisper_models_dir, model_name)
            model_path_or_name = model_dir if os.path.isdir(model_dir) else model_name

            logger.info(
                "Loading model '%s' (device=%s, compute_type=%s, root=%s)...",
                model_name,
                settings.whisper_device,
                settings.effective_compute_type,
                settings.whisper_models_dir,
            )

            try:
                model = WhisperModel(
                    model_path_or_name,
                    device=settings.whisper_device,
                    compute_type=settings.effective_compute_type,
                    download_root=settings.whisper_models_dir,
                )
                self._models[model_name] = model
                logger.info("Successfully loaded model '%s'", model_name)
            except Exception as e:
                logger.error("Failed to load model '%s': %s", model_name, str(e))
                raise

        self._is_loaded = True

    def get_model(self, requested_model: str) -> WhisperModel:
        resolved = settings.resolve_model(requested_model)
        if resolved not in self._models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": f"The model '{requested_model}' does not exist or is not loaded.",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "model_not_found",
                    }
                },
            )
        return self._models[resolved]

    def _sync_transcribe(
        self,
        model: WhisperModel,
        audio_path: str,
        task: str,
        language: Optional[str],
        prompt: Optional[str],
        temperature: Optional[float],
        word_timestamps: bool,
    ) -> Tuple[str, str, float, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Synchronous inference executed inside a worker thread."""
        kwargs: Dict[str, Any] = {
            "task": task,
            "beam_size": settings.whisper_beam_size,
            "vad_filter": settings.whisper_vad,
            "word_timestamps": word_timestamps,
        }

        if language:
            kwargs["language"] = language.lower().strip()
        if prompt:
            kwargs["initial_prompt"] = prompt
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        segments_iter, info = model.transcribe(audio_path, **kwargs)

        segments_list: List[Dict[str, Any]] = []
        words_list: List[Dict[str, Any]] = []

        for seg in segments_iter:
            segments_list.append(
                {
                    "id": seg.id,
                    "seek": seg.seek,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text,
                    "tokens": seg.tokens,
                    "temperature": seg.temperature,
                    "avg_logprob": round(seg.avg_logprob, 4),
                    "compression_ratio": round(seg.compression_ratio, 4),
                    "no_speech_prob": round(seg.no_speech_prob, 4),
                }
            )
            if seg.words:
                for w in seg.words:
                    words_list.append(
                        {
                            "word": w.word,
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                        }
                    )

        full_text = "".join(s["text"] for s in segments_list).strip()
        detected_lang = info.language or (language or "en")
        duration = info.duration or 0.0

        return full_text, detected_lang, duration, segments_list, words_list

    async def transcribe(
        self,
        requested_model: str,
        audio_path: str,
        task: str = "transcribe",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        word_timestamps: bool = False,
    ) -> Tuple[str, str, float, List[Dict[str, Any]], List[Dict[str, Any]]]:
        model = self.get_model(requested_model)
        return await asyncio.to_thread(
            self._sync_transcribe,
            model,
            audio_path,
            task,
            language,
            prompt,
            temperature,
            word_timestamps,
        )


engine = WhisperEngine()
