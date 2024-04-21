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
    """
    A class used to represent an Authorization Header.

    ...

    Attributes
    ----------
    name : str
        a string representing the name of the header. It is excluded from the model's output. (default is a unique uuid)

    Methods
    -------
    build_default_header():
        Returns a default AuthHeader object.

    get_auth_json(auth_obj: dict):
        Returns a tuple (auth_dict, is_query_param) where auth_dict is a dictionary representation of the AuthHeader object 
        and is_query_param is a boolean indicating whether the authorization type is 'query_param' or not.
    """
    name: str = Field(description="name", default=str(uuid.uuid4()), exclude=True)

    @staticmethod
    def build_default_header():
        return AuthHeader()

    def get_auth_json(self, auth_obj: dict):
        """
        Method to get a dictionary representation of the AuthHeader object and a boolean indicating whether the authorization 
        type is 'query_param' or not.

        Parameters
        ----------
        auth_obj : dict
            a dictionary containing authorization details

        Returns
        -------
        tuple
            a tuple (auth_dict, is_query_param) where auth_dict is a dictionary representation of the AuthHeader object 
            and is_query_param is a boolean indicating whether the authorization type is 'query_param' or not.
        """
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
    """
    A class used to represent an Openplugin Response.

    ...

    Attributes
    ----------
    default_output_module : str
        a string representing the default output module.

    output_module_map : Dict[str, Response]
        a dictionary mapping strings to Response objects.

    Methods
    -------
    get_default_output_module_response():
        Returns the Response object associated with the default output module.
    """

    default_output_module: str
    output_module_map: Dict[str, Response]

    def get_default_output_module_response(self):
        """
        Method to get the Response object associated with the default output module.

        Returns
        -------
        Response
            the Response object associated with the default output module
        """
        return self.output_module_map.get(self.default_output_module)


class LLM(BaseModel):
    """
    A class used to represent a Language Learning Model (LLM).

    ...

    Attributes
    ----------
    provider : str
        a string representing the LLM provider. (default is "OpenAI")

    model : str
        a string representing the LLM model name. (default is "gpt-3.5-turbo-0613")

    frequency_penalty : float
        a float representing the LLM frequency penalty. (default is 0)

    max_tokens : int
        an integer representing the LLM max tokens. (default is 2048)

    presence_penalty : float
        a float representing the LLM presence penalty. (default is 0)

    temperature : float
        a float representing the LLM temperature. (default is 0)

    top_p : float
        a float representing the LLM top_p. (default is 1)

    Methods
    -------
    build_default_llm():
        Returns a default LLM object.
    """
 
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
        """
        Static method to build a default LLM object.

        Returns
        -------
        LLM
            a default LLM object
        """
        return LLM()


class Approach(BaseModel):
    """
    A class used to represent an Approach.

    ...

    Attributes
    ----------
    name : str
        a string representing the name of the approach. (default is a unique uuid)

    base_strategy : str
        a string representing the base strategy. (default is "oai functions")

    pre_prompt : Optional[str]
        an optional string representing the pre prompt. (default is None)

    llm : LLM
        an instance of the LLM class.

    Methods
    -------
    build_default_approach():
        Returns a default Approach object.
    """
    name: str = Field(description="Approach name", default=str(uuid.uuid4()))
    base_strategy: str = Field(description="Base strategy", default="oai functions")
    pre_prompt: Optional[str] = Field(description="pre prompt", default=None)
    llm: LLM

    @staticmethod
    def build_default_approach():
        """
        Static method to build a default Approach object.

        Returns
        -------
        Approach
            a default Approach object
        """
        return Approach(llm=LLM.build_default_llm())


PLUGIN_EXECUTION_API_PATH = "/api/plugin-execution-pipeline"
PLUGIN_SELECTOR_API_PATH = "/api/plugin-selector"


class OpenpluginService(BaseModel):
    """
    A class used to represent an Openplugin Service.

    ...

    Attributes
    ----------
    openplugin_server_endpoint : str
        a string representing the openplugin server endpoint.

    openplugin_api_key : SecretStr
        a secret string representing the openplugin API key. This field is excluded from the model's output.

    client : Any
        an instance of the httpx.Client class.

    Methods
    -------
    ping():
        Returns a string indicating whether the client was able to successfully get a response from the "/api/info" endpoint.

    remote_server_version():
        Returns a string representing the version of the remote server.

    run(openplugin_manifest_url: str, prompt: str, conversation: List[str], header: AuthHeader, approach: Approach, config: Config, output_module_names: List[str]):
        Returns a Response object representing the result of a POST request to the PLUGIN_EXECUTION_API_PATH endpoint.

    select_a_plugin(openplugin_manifest_urls: List[str], prompt: str, conversation: List[str], pipeline_name: str, config: Config, llm: LLM):
        Returns a string representing the detected plugin from a POST request to the PLUGIN_SELECTOR_API_PATH endpoint.
    """
     
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
        """
        Method to check if the client is able to successfully get a response from the "/api/info" endpoint.

        Returns
        -------
        str
            a string indicating whether the client was able to successfully get a response from the "/api/info" endpoint.
        """
        result = self.client.get("/api/info")
        if result.status_code == 200:
            return "success"
        return "failed"

    def remote_server_version(self) -> str:
        """
        Method to get the version of the remote server.

        Returns
        -------
        str
            a string representing the version of the remote server.
        """
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
        """
        Method to make a POST request to the PLUGIN_EXECUTION_API_PATH endpoint.

        Parameters
        ----------
        openplugin_manifest_url : str
            a string representing the openplugin manifest URL.

        prompt : str
            a string representing the prompt.

        conversation : List[str]
            a list of strings representing the conversation. (default is [])

        header : AuthHeader
            an instance of the AuthHeader class. (default is a default AuthHeader object)

        approach : Approach
            an instance of the Approach class. (default is a default Approach object)

        config : Config
            an instance of the Config class. (default is a Config object)

        output_module_names : List[str]
            a list of strings representing the output module names. (default is [])

        Returns
        -------
        Response
            a Response object representing the result of the POST request.
        """
            
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

    def select_a_plugin(
        self,
        openplugin_manifest_urls: List[str],
        prompt: str,
        conversation: List[str] = [],
        pipeline_name: str = "oai functions",
        config: Config = Config(),
        llm: LLM = LLM.build_default_llm(),
    ) -> str:
        """
        Method to select a plugin by making a POST request to the PLUGIN_SELECTOR_API_PATH endpoint.

        Parameters
        ----------
        openplugin_manifest_urls : List[str]
            a list of strings representing the openplugin manifest URLs.

        prompt : str
            a string representing the prompt.

        conversation : List[str]
            a list of strings representing the conversation. (default is [])

        pipeline_name : str
            a string representing the pipeline name. (default is "oai functions")

        config : Config
            an instance of the Config class. (default is a Config object)

        llm : LLM
            an instance of the LLM class. (default is a default LLM object)

        Returns
        -------
        str
            a string representing the detected plugin.
        """
            
        llm_dict = llm.dict(exclude_none=True)
        llm_dict["model_name"]=llm_dict.pop("model")
        payload = json.dumps(
            {
                "messages": [{"content": prompt, "message_type": "HumanMessage"}],
                "pipeline_name": pipeline_name,
                "openplugin_manifest_urls": openplugin_manifest_urls,
                "config": config.dict(exclude_none=True),
                "llm": llm_dict,
            }
        )
        result = self.client.post(PLUGIN_SELECTOR_API_PATH, data=payload, timeout=30)
        if result.status_code != 200:
            raise Exception(
                f"Failed to run openplugin service. Status code: {result.status_code}, Reason: {result.text}"
            )
        response_json = result.json()
        return response_json.get("detected_plugin")

    class Config:
        arbitrary_types_allowed = False


def get_output_module_names(openplugin_manifest_url: str) -> List[str]:
    """
    Function to get the names of the output modules from an openplugin manifest URL.

    Parameters
    ----------
    openplugin_manifest_url : str
        a string representing the openplugin manifest URL.

    Returns
    -------
    List[str]
        a list of strings representing the names of the output modules.
    """
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
