"""Chat model selection.

Models are named `provider:model`, as accepted by `init_chat_model`
(https://docs.langchain.com/oss/python/deepagents/models). The extra
`lmstudio:` prefix targets LM Studio's OpenAI-compatible server
(https://lmstudio.ai/docs/developer/openai-compat).
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

LM_STUDIO_PREFIX = "lmstudio:"


def build_chat_model(model: str, lm_studio_url: str) -> BaseChatModel:
    if model.startswith(LM_STUDIO_PREFIX):
        return init_chat_model(
            model.removeprefix(LM_STUDIO_PREFIX),
            model_provider="openai",
            base_url=lm_studio_url,
            api_key="lm-studio",
            use_responses_api=False,
            temperature=0.3,
        )
    if ":" not in model:
        raise ValueError(
            f"Model {model!r} has no provider prefix; use e.g. "
            "google_genai:gemini-2.5-flash or lmstudio:<model-id>"
        )
    return init_chat_model(model, temperature=0.3)
