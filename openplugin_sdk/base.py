import json
import uuid
import httpx
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field, SecretStr


class Response(BaseModel):
    name: str
    data_type: str
    mime_type: Optional[str] = None
    value: Any


class OpenpluginResponse(BaseModel):
    default_output_module: str
    output_module_map: Dict[str, Response]

    def get_default_output_module_response(self):
        return self.output_module_map.get(self.default_output_module)


class LLM(BaseModel):
    provider: str = Field(description="LLM provider", default="OpenAI")
    model_name: str = Field(
        description="LLM model name", alias="model_name",default="gpt-3.5-turbo-0613"
    )
    frequency_penalty: float = Field(description="LLM frequency penalty", default=0)
    max_tokens: int = Field(description="LLM max tokens", default=2048)
    presence_penalty: float = Field(description="LLM presence penalty", default=0)
    temperature: float = Field(description="LLM temperature", default=0)
    top_p: float = Field(description="LLM top_p", default=1)

    @staticmethod
    def build_default_llm():
        return LLM()


class Approach(BaseModel):
    name: str = Field(description="Approach name", default=str(uuid.uuid4()))
    base_strategy: str = Field(description="Base strategy", default="oai functions")
    pre_prompt: Optional[str] = Field(description="pre prompt", default=None)
    llm: LLM

    @staticmethod
    def build_default_approach():
        return Approach(llm=LLM.build_default_llm())


class Header(BaseModel):
    user_http_token: Optional[str] = Field(description="Field 1", default=None)

    @staticmethod
    def build_default_header():
        return Header()


PLUGIN_EXECUTION_API_PATH = "/api/plugin-execution-pipeline"


class OpenpluginService(BaseModel):
    remote_server_endpoint: str = Field(..., description="Field 1")
    api_key: SecretStr = Field(..., description="Field 2", exclude=True)
    client: Any = None  # httpx.Client

    def __init__(self, **data):
        super().__init__(**data)
        self.client = httpx.Client(
            base_url=self.remote_server_endpoint,
            headers={"x-api-key": self.api_key.get_secret_value(), "Content-Type": "application/json"},
        )

    def __del__(self):
        if self.client:
            self.client.close()

    def run(
        self,
        openplugin_manifest_url: str,
        prompt: str,
        conversation: List[str] = [],
        header: Header = Header.build_default_header(),
        approach: Approach = Approach.build_default_approach(),
        output_module_names: List[str] = [],
    ) -> Response:
        payload = json.dumps(
            {
                "prompt": prompt,
                "conversation": conversation,
                "openplugin_manifest_url": openplugin_manifest_url,
                "header": header.dict(),
                "approach": approach.dict(),
                "output_module_names": output_module_names,
            }
        )
        result = self.client.post(PLUGIN_EXECUTION_API_PATH, data=payload)
        response_json = result.json()
        openplugin_response = OpenpluginResponse(
            default_output_module=response_json.get("response").get(
                "default_output_module"
            ),
            output_module_map=response_json.get("response").get("output_module_map"),
        )
        if len(output_module_names) == 1:
            return openplugin_response.output_module_map.get(output_module_names[0])
        return openplugin_response.get_default_output_module_response()

    class Config:
        arbitrary_types_allowed = False

def get_output_module_names(openplugin_manifest_url:str) -> List[str]:
    response=httpx.get(openplugin_manifest_url)
    if response.status_code!=200:
        raise Exception(f"Failed to fetch openplugin manifest from {openplugin_manifest_url}")
    response_json=response.json()
    names=[]
    for output_module in response_json.get("output_modules"):
        names.append(output_module.get("name"))
    return names
