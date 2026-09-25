import json

from datacontract.ai.pii_detector import PiiDetector
from datacontract.model.exceptions import DataContractException

_MODEL = "claude-haiku-4-5-20251001"

_PROMPT_TEMPLATE = (
    "For each of the following database column names, decide whether it is likely to contain "
    "personally identifiable information (PII) such as names, emails, phone numbers, addresses, "
    "government IDs, or other data that identifies an individual person. "
    "Respond with ONLY a JSON object mapping each column name to true or false, no other text.\n\n"
    "Column names: {column_names}"
)


class AnthropicPiiDetector(PiiDetector):
    """Classifies column names as PII by calling the Anthropic API directly."""

    def detect(self, column_names: list[str]) -> dict[str, bool]:
        try:
            import anthropic
        except ImportError as e:
            raise DataContractException(
                type="anthropic-connection",
                name="anthropic extra missing",
                reason="Install the extra datacontract-cli[ai] to use --pii-detector anthropic",
                engine="datacontract-cli",
                original_exception=e,
            )

        api_key = self.config.get_anthropic_api_key(required=True)
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _PROMPT_TEMPLATE.format(column_names=json.dumps(column_names))
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            result = json.loads(response.content[0].text)
            return {name: bool(result.get(name, False)) for name in column_names}
        except Exception as e:
            raise DataContractException(
                type="anthropic-connection",
                name="anthropic pii detection failed",
                reason=f"Could not classify columns via the Anthropic API: {e}",
                engine="datacontract-cli",
                original_exception=e,
            )
