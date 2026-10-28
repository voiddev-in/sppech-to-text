import math
from typing import Any, Dict, List, Optional

# Standard Whisper ISO-639-1 code to language name map
LANGUAGE_CODES_TO_NAMES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}


def resolve_language_name(lang_code_or_name: Optional[str]) -> str:
    if not lang_code_or_name:
        return "english"
    code = lang_code_or_name.lower().strip()
    return LANGUAGE_CODES_TO_NAMES.get(code, code)


def format_timestamp(seconds: float, separator: str = ".") -> str:
    """Formats seconds into HH:MM:SS.mmm (VTT) or HH:MM:SS,mmm (SRT)."""
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def to_srt(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_str = format_timestamp(seg["start"], separator=",")
        end_str = format_timestamp(seg["end"], separator=",")
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start_str} --> {end_str}\n{text}\n")
    return "\n".join(lines).strip() + "\n" if lines else ""


def to_vtt(segments: List[Dict[str, Any]]) -> str:
    cues = []
    for seg in segments:
        start_str = format_timestamp(seg["start"], separator=".")
        end_str = format_timestamp(seg["end"], separator=".")
        text = seg.get("text", "").strip()
        cues.append(f"{start_str} --> {end_str}\n{text}")
    if not cues:
        return "WEBVTT\n"
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


def format_verbose_json(
    task: str,
    language: str,
    duration: float,
    text: str,
    segments: List[Dict[str, Any]],
    words: Optional[List[Dict[str, Any]]] = None,
    include_segments: bool = True,
    include_words: bool = False,
) -> Dict[str, Any]:
    duration_rounded = round(float(duration), 2)
    result: Dict[str, Any] = {
        "task": task,
        "language": resolve_language_name(language),
        "duration": duration_rounded,
        "text": text,
        "usage": {
            "type": "duration",
            "seconds": math.ceil(duration_rounded),
        },
    }

    if include_segments:
        result["segments"] = segments

    if include_words and words is not None:
        result["words"] = words

    return result
