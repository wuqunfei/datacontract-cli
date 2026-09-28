import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

from datacontract.ai.pii_detector import PiiDetector
from datacontract.model.exceptions import DataContractException

_PROMPT_TEMPLATE = (
    "For each of the following database column names, decide whether it is likely to contain "
    "personally identifiable information (PII) such as names, emails, phone numbers, addresses, "
    "government IDs, or other data that identifies an individual person. "
    "Respond with ONLY a JSON object mapping each column name to true or false, no other text.\n\n"
    "Column names: {column_names}"
)


class DatabricksPiiDetector(PiiDetector):
    """Classifies column names as PII by calling a self-hosted Claude model behind a Databricks
    Model Serving endpoint, using the same workspace credentials as `import databricks`."""

    def detect(self, column_names: list[str]) -> dict[str, bool]:
        endpoint = self.config.get_databricks_pii_endpoint(required=True)
        profile = self.config.get_databricks_profile()
        if profile:
            workspace_client = WorkspaceClient(profile=profile)
        else:
            workspace_client = WorkspaceClient(
                host=self.config.get_databricks_server_hostname(required=True),
                token=self.config.get_databricks_token(required=True),
            )

        prompt = _PROMPT_TEMPLATE.format(column_names=json.dumps(column_names))
        try:
            response = workspace_client.serving_endpoints.query(
                name=endpoint,
                messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
            )
            result = json.loads(response.choices[0].message.content)
            return {name: bool(result.get(name, False)) for name in column_names}
        except DataContractException:
            raise
        except Exception as e:
            raise DataContractException(
                type="databricks-connection",
                name="databricks pii detection failed",
                reason=f"Could not classify columns via the Databricks serving endpoint '{endpoint}': {e}",
                engine="datacontract-cli",
                original_exception=e,
            )
