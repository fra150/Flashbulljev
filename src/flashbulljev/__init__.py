"""flashbulljev - fast and stored System One engine."""
__version__ = "0.3.0"
__model__ = "flashbulljev"

from .backends import FakeBackend, OllamaBackend, get_backend
from .calibration import apply_temperature, brier, ece, fit_temperature, nll
from .engine import FlashBullJevEngine, decide_many, decide_many_real
from .memory_gf import (
    FragmentMemory,
    MemoriaGF,
    certifica_frammento,
    certify_fragment,
    verifica_quiete,
    verify_quiescence,
)
from .pipeline import run_pipeline, PipelineResult
from .prompts import build_boolean_prompt, build_mc_prompt
from .schemas import (
    BooleanQuestion,
    ChoiceQuestion,
    NumericQuestion,
    ScoreQuestion,
    parse_questions,
)

__all__ = [
    "__version__",
    "__model__",
    "FlashBullJevEngine",
    "decide_many",
    "decide_many_real",
    "FragmentMemory",
    "MemoriaGF",
    "verify_quiescence",
    "verifica_quiete",
    "certify_fragment",
    "certifica_frammento",
    "run_pipeline",
    "PipelineResult",
    "BooleanQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "NumericQuestion",
    "parse_questions",
    "FakeBackend",
    "OllamaBackend",
    "get_backend",
    "apply_temperature",
    "fit_temperature",
    "nll",
    "brier",
    "ece",
    "build_mc_prompt",
    "build_boolean_prompt",
]
