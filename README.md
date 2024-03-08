# OpenPluginSDK

The OpenPlugin SDK for python is a powerful and versatile toolkit designed to streamline the integration and consumption of OpenPlugin API services. This SDK empowers developers to effortlessly leverage the capabilities of the OpenPlugin ecosystem, promoting rapid development.

Currently available in preview mode. Try it out today!


## Getting started

1. Install the package from pypi:

```
pip install openplugin-sdk
```

Note: You need openplugin service API to run openplugin sdk

2. Setup openplugin service:

```
remote_server_endpoint = "...."
openplugin_api_key = "...."
svc = OpenpluginService(
        remote_server_endpoint=remote_server_endpoint, api_key=openplugin_api_key
)
```

3. Run openplugin

```
openplugin_manifest_url = "...."
prompt = "..."
output_module_name="..."

response = svc.run(
        openplugin_manifest_url=openplugin_manifest_url,
        prompt=prompt,
        output_module_names=[output_module_name],
)
print(f"Response={response.value}")
```

## Starting openplugin service locally

1. Install the package from pypi:

```
pip install openplugin
```

2. Start openplugin service

```
export OPENAI_API_KEY=<your key>
openplugin start-server
```

Note: Learn more about openplugin at: https://openplugin.org/


## Examples

Several samples to run an openplugin can be found in the examples folder

- [Simple Demo](examples/simple_demo.ipynb)
- [User Auth Demo](examples/advanced_demo.ipynb)
- [Advance Demo](examples/user_auth_demo.ipynb)

