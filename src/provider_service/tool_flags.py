"""vLLM tool-call parser flags for InferX serve commands."""

from __future__ import annotations

DEFAULT_TOOL_CALL_PARSER = "hermes"

# Repo-id needles. More specific entries must appear first.
MODEL_NAME_TOOL_PARSERS = (
    ("qwen3.5", "qwen3_xml"),
    ("qwen3_5", "qwen3_xml"),
    ("qwen3-5", "qwen3_xml"),
    ("qwen3-coder", "qwen3_xml"),
    ("qwen3_coder", "qwen3_xml"),
    ("qwen3coder", "qwen3_xml"),
    ("qwen3", "hermes"),
    ("qwen2.5", "hermes"),
    ("qwen2", "hermes"),
    ("deepseek-v3.2", "deepseek_v32"),
    ("deepseek-v3.1", "deepseek_v31"),
    ("deepseek-v3", "deepseek_v3"),
    ("deepseek-v4", "deepseek_v4"),
    ("llama-4", "llama4_pythonic"),
    ("llama4", "llama4_pythonic"),
    ("llama-3.2", "pythonic"),
    ("llama-3", "llama3_json"),
    ("llama3", "llama3_json"),
    ("mixtral", "mistral"),
    ("mistral", "mistral"),
    ("gigachat3.5", "gigachat3"),
    ("gigachat-3", "gigachat3"),
    ("gigachat3", "gigachat3"),
    ("gigachat", "gigachat3"),
    ("minicpm5", "minicpm5"),
    ("minimax-m3", "minimax_m3"),
    ("minimax_m3", "minimax_m3"),
    ("kimi-k2.5", "kimi_k2"),
    ("kimi-k2", "kimi_k2"),
    ("kimi_k2", "kimi_k2"),
    ("minimax-m2", "minimax_m2"),
    ("minimax", "minimax"),
    ("gemma-4", "gemma4"),
    ("functiongemma", "functiongemma"),
    ("gemma-3", "gemma4"),
    ("hermes-3", "hermes"),
    ("hermes-2", "hermes"),
    ("hermes3", "hermes"),
    ("hermes2", "hermes"),
    ("hermes", "hermes"),
    ("phi-4-mini", "phi4_mini_json"),
    ("phi4-mini", "phi4_mini_json"),
    ("xlam", "xlam"),
    ("longcat", "longcat"),
    ("hunyuan", "hunyuan_a13b"),
    ("apertus", "apertus"),
    ("jamba", "jamba"),
    ("ernie-4.5", "ernie45"),
    ("ernie4.5", "ernie45"),
    ("ernie4_5", "ernie45"),
    ("ernie", "ernie45"),
    ("seed-oss", "seed_oss"),
    ("seed_oss", "seed_oss"),
    ("step3p5", "step3p5"),
    ("step-3.5", "step3p5"),
    ("step3", "step3"),
    ("step-3", "step3"),
    ("poolside", "poolside_v1"),
    ("hy-v3", "hy_v3"),
    ("mimo", "mimo"),
    ("toolace", "pythonic"),
    ("ultravox", "pythonic"),
)


def tool_call_parser_for_model(hf_model: str) -> str:
    needle = str(hf_model or "").strip().lower()
    for token, parser in MODEL_NAME_TOOL_PARSERS:
        if token in needle:
            return parser
    return DEFAULT_TOOL_CALL_PARSER


def vllm_auto_tool_choice_flags(hf_model: str) -> list[str]:
    """Argv that makes stock vLLM accept OpenAI tools and tool_choice=auto."""
    return [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_call_parser_for_model(hf_model),
    ]
