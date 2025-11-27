from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseLLM(ABC):
    def __init__(self, model_name: str, model_params: Optional[dict[str, Any]] = None, **kwargs: Any):
        self.model_name = model_name
        self.model_params = model_params or {}

    @abstractmethod
    def predict(self, input: str) -> str:
        """"""


from openai import OpenAI

class LLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.hunyuan.cloud.tencent.com/v1", model_params: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(model_name, model_params, **kwargs)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def predict(self, input: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": "input"}]
        )
        return response.choices[0].message.content


llm = LLM(model_name="hunyuan-lite", api_key="")

print(llm.predict("你好"))



