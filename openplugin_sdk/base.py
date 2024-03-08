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


class Config(BaseModel):
    """
    Represents the API configuration for a plugin.
    """

    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    google_palm_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region_name: Optional[str] = None
    azure_api_key: Optional[str] = None



class AuthHeader(BaseModel):
    name: str = Field(description="name", default=str(uuid.uuid4()), exclude=True)

    @staticmethod
    def build_default_header():
        return AuthHeader()

    def get_auth_json(self, auth_obj: dict):
        is_query_param = False
        auth_dict = self.dict()
        if auth_obj.get("authorization_type") == "query_param":
            is_query_param = True
            query_param_key = auth_obj.get("query_param_key")
            if query_param_key and "user_http_token" in auth_dict:
                auth_dict[query_param_key] = auth_dict.get("user_http_token")
                auth_dict.pop("user_http_token")
        return auth_dict, is_query_param


class UserAuthHeader(AuthHeader):
    user_http_token: str = Field(description="User http token")


class OpenpluginResponse(BaseModel):
    default_output_module: str
    output_module_map: Dict[str, Response]

    def get_default_output_module_response(self):
        return self.output_module_map.get(self.default_output_module)


class LLM(BaseModel):
    provider: str = Field(description="LLM provider", default="OpenAI")
    model: str = Field(
        description="LLM model name",
        alias="model_name",
        default="gpt-3.5-turbo-0613",
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


PLUGIN_EXECUTION_API_PATH = "/api/plugin-execution-pipeline"


class OpenpluginService(BaseModel):
    openplugin_server_endpoint: str = Field(..., description="Field 1")
    openplugin_api_key: SecretStr = Field(..., description="Field 2", exclude=True)
    client: Any = None  # httpx.Client

    def __init__(self, **data):
        super().__init__(**data)
        self.client = httpx.Client(
            base_url=self.openplugin_server_endpoint,
            headers={
                "x-api-key": self.openplugin_api_key.get_secret_value(),
                "Content-Type": "application/json",
            },
        )

    def __del__(self):
        if self.client:
            self.client.close()

    def ping(self) -> str:
        result = self.client.get("/api/info")
        if result.status_code == 200:
            return "success"
        return "failed"

    def remote_server_version(self) -> str:
        result = self.client.get("/api/info")
        if result.status_code == 200:
            return result.json().get("version")
        return "failed"

    def run(
        self,
        openplugin_manifest_url: str,
        prompt: str,
        conversation: List[str] = [],
        header: AuthHeader = AuthHeader.build_default_header(),
        approach: Approach = Approach.build_default_approach(),
        config: Config = Config(),
        output_module_names: List[str] = [],
    ) -> Response:
        openplugin_manifest_json = httpx.get(openplugin_manifest_url).json()
        auth_dict, is_query_param = header.get_auth_json(
            openplugin_manifest_json.get("auth")
        )
        if is_query_param:
            header_dict = {}
            query_param_dict = auth_dict
        else:
            header_dict = auth_dict
            query_param_dict = {}
        payload = json.dumps(
            {
                "prompt": prompt,
                "conversation": conversation,
                "openplugin_manifest_url": openplugin_manifest_url,
                "header": header_dict,
                "config": config.dict(exclude_none=True),
                "auth_query_param": query_param_dict,
                "approach": approach.dict(by_alias=True),
                "output_module_names": output_module_names,
            }
        )
        result = self.client.post(
            PLUGIN_EXECUTION_API_PATH, data=payload, timeout=30
        )
        if result.status_code != 200:
            raise Exception(
                f"Failed to run openplugin service. Status code: {result.status_code}, Reason: {result.text}"
            )

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


def get_output_module_names(openplugin_manifest_url: str) -> List[str]:
    response = httpx.get(openplugin_manifest_url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch openplugin manifest from {openplugin_manifest_url}"
        )
    response_json = response.json()
    names = []
    for output_module in response_json.get("output_modules"):
        names.append(output_module.get("name"))
    return names
