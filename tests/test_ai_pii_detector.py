import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datacontract.ai.annotate import mark_pii_columns
from datacontract.ai.pii_detector import PiiDetector
from datacontract.ai.pii_detector_factory import pii_detector_factory
from datacontract.ai.rule_based_pii_detector import RuleBasedPiiDetector
from datacontract.config import Config
from datacontract.imports.odcs_helper import create_odcs, create_property, create_schema_object
from datacontract.model.exceptions import DataContractException


def test_rule_based_detector_matches_known_pii_column_names():
    detector = RuleBasedPiiDetector()

    result = detector.detect(
        ["email", "customer_email", "SSN", "user_ssn", "phone_number", "first_name", "home_address", "date_of_birth"]
    )

    assert result == {
        "email": True,
        "customer_email": True,
        "SSN": True,
        "user_ssn": True,
        "phone_number": True,
        "first_name": True,
        "home_address": True,
        "date_of_birth": True,
    }


def test_rule_based_detector_does_not_match_non_pii_column_names():
    detector = RuleBasedPiiDetector()

    result = detector.detect(["order_id", "quantity", "created_at", "is_active", "telephone_book_id"])

    assert result == {
        "order_id": False,
        "quantity": False,
        "created_at": False,
        "is_active": False,
        "telephone_book_id": False,
    }


class _StubDetector(PiiDetector):
    def __init__(self, matches: set):
        super().__init__(config=None)
        self.matches = matches
        self.calls: list = []

    def detect(self, column_names: list[str]) -> dict[str, bool]:
        self.calls.append(list(column_names))
        return {name: name in self.matches for name in column_names}


def _odcs_with_properties(properties):
    schema_obj = create_schema_object(name="orders", properties=properties)
    odcs = create_odcs()
    odcs.schema_ = [schema_obj]
    return odcs, schema_obj


def test_mark_pii_columns_sets_classification_and_critical_flag():
    email = create_property(name="email", logical_type="string")
    order_id = create_property(name="order_id", logical_type="string")
    odcs, _ = _odcs_with_properties([email, order_id])
    detector = _StubDetector(matches={"email"})

    mark_pii_columns(odcs, detector)

    assert email.classification == "PII"
    assert email.criticalDataElement is True
    assert order_id.classification is None
    assert order_id.criticalDataElement is None


def test_mark_pii_columns_never_overwrites_existing_classification():
    email = create_property(name="email", logical_type="string", classification="public")
    odcs, _ = _odcs_with_properties([email])
    detector = _StubDetector(matches={"email"})

    mark_pii_columns(odcs, detector)

    assert email.classification == "public"


def test_mark_pii_columns_skips_detector_call_when_all_columns_classified():
    email = create_property(name="email", logical_type="string", classification="public")
    odcs, _ = _odcs_with_properties([email])
    detector = _StubDetector(matches={"email"})

    mark_pii_columns(odcs, detector)

    assert detector.calls == []


def test_mark_pii_columns_recurses_into_struct_properties():
    nested_email = create_property(name="email", logical_type="string")
    contact = create_property(name="contact", logical_type="object", properties=[nested_email])
    odcs, _ = _odcs_with_properties([contact])
    detector = _StubDetector(matches={"email"})

    mark_pii_columns(odcs, detector)

    assert nested_email.classification == "PII"


def test_factory_creates_rule_based_detector():
    detector = pii_detector_factory.create("rule")

    assert isinstance(detector, RuleBasedPiiDetector)


def test_factory_raises_for_unknown_detector_name():
    with pytest.raises(ValueError, match="not supported"):
        pii_detector_factory.create("does-not-exist")


def test_anthropic_detector_parses_response(monkeypatch):
    from datacontract.ai.anthropic_pii_detector import AnthropicPiiDetector

    fake_response = SimpleNamespace(content=[SimpleNamespace(text='{"email": true, "order_id": false}')])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    fake_anthropic_module = SimpleNamespace(Anthropic=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

    detector = AnthropicPiiDetector(Config(anthropic_api_key="key"))
    result = detector.detect(["email", "order_id"])

    assert result == {"email": True, "order_id": False}
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert "email" in call_kwargs["messages"][0]["content"]
    assert "order_id" in call_kwargs["messages"][0]["content"]


def test_anthropic_detector_requires_api_key(monkeypatch):
    from datacontract.ai.anthropic_pii_detector import AnthropicPiiDetector

    monkeypatch.delenv("DATACONTRACT_ANTHROPIC_API_KEY", raising=False)
    detector = AnthropicPiiDetector(Config.model_construct())

    with pytest.raises(DataContractException, match="DATACONTRACT_ANTHROPIC_API_KEY"):
        detector.detect(["email"])


def test_anthropic_detector_raises_when_package_missing(monkeypatch):
    from datacontract.ai.anthropic_pii_detector import AnthropicPiiDetector

    monkeypatch.setitem(sys.modules, "anthropic", None)
    detector = AnthropicPiiDetector(Config(anthropic_api_key="key"))

    with pytest.raises(DataContractException, match=r"datacontract-cli\[ai\]"):
        detector.detect(["email"])


def test_anthropic_detector_raises_on_malformed_response(monkeypatch):
    from datacontract.ai.anthropic_pii_detector import AnthropicPiiDetector

    fake_response = SimpleNamespace(content=[SimpleNamespace(text="not json")])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    fake_anthropic_module = SimpleNamespace(Anthropic=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

    detector = AnthropicPiiDetector(Config(anthropic_api_key="key"))

    with pytest.raises(DataContractException, match="Could not classify columns"):
        detector.detect(["email"])


def test_anthropic_detector_raises_on_non_dict_response(monkeypatch):
    from datacontract.ai.anthropic_pii_detector import AnthropicPiiDetector

    fake_response = SimpleNamespace(content=[SimpleNamespace(text='["email", "order_id"]')])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    fake_anthropic_module = SimpleNamespace(Anthropic=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

    detector = AnthropicPiiDetector(Config(anthropic_api_key="key"))

    with pytest.raises(DataContractException, match="Could not classify columns"):
        detector.detect(["email"])


def test_databricks_detector_queries_serving_endpoint(monkeypatch):
    from datacontract.ai.databricks_pii_detector import DatabricksPiiDetector

    fake_message = SimpleNamespace(content='{"email": true, "order_id": false}')
    fake_choice = SimpleNamespace(message=fake_message)
    fake_response = SimpleNamespace(choices=[fake_choice])
    fake_serving_endpoints = MagicMock()
    fake_serving_endpoints.query.return_value = fake_response
    fake_workspace_client = MagicMock(serving_endpoints=fake_serving_endpoints)
    monkeypatch.setattr(
        "datacontract.ai.databricks_pii_detector.WorkspaceClient",
        MagicMock(return_value=fake_workspace_client),
    )

    config = Config(databricks_pii_endpoint="pii-endpoint", databricks_profile="DEFAULT")
    detector = DatabricksPiiDetector(config)
    result = detector.detect(["email", "order_id"])

    assert result == {"email": True, "order_id": False}
    call_kwargs = fake_serving_endpoints.query.call_args.kwargs
    assert call_kwargs["name"] == "pii-endpoint"


def test_databricks_detector_requires_endpoint(monkeypatch):
    from datacontract.ai.databricks_pii_detector import DatabricksPiiDetector

    monkeypatch.delenv("DATACONTRACT_DATABRICKS_PII_ENDPOINT", raising=False)
    config = Config(databricks_profile="DEFAULT")
    detector = DatabricksPiiDetector(config)

    with pytest.raises(DataContractException, match="DATACONTRACT_DATABRICKS_PII_ENDPOINT"):
        detector.detect(["email"])


def test_databricks_detector_raises_on_malformed_response(monkeypatch):
    from datacontract.ai.databricks_pii_detector import DatabricksPiiDetector

    fake_message = SimpleNamespace(content="not json")
    fake_choice = SimpleNamespace(message=fake_message)
    fake_response = SimpleNamespace(choices=[fake_choice])
    fake_serving_endpoints = MagicMock()
    fake_serving_endpoints.query.return_value = fake_response
    fake_workspace_client = MagicMock(serving_endpoints=fake_serving_endpoints)
    monkeypatch.setattr(
        "datacontract.ai.databricks_pii_detector.WorkspaceClient",
        MagicMock(return_value=fake_workspace_client),
    )

    config = Config(databricks_pii_endpoint="pii-endpoint", databricks_profile="DEFAULT")
    detector = DatabricksPiiDetector(config)

    with pytest.raises(DataContractException, match="Could not classify columns"):
        detector.detect(["email"])


def test_databricks_detector_raises_on_non_dict_response(monkeypatch):
    from datacontract.ai.databricks_pii_detector import DatabricksPiiDetector

    fake_message = SimpleNamespace(content='["email", "order_id"]')
    fake_choice = SimpleNamespace(message=fake_message)
    fake_response = SimpleNamespace(choices=[fake_choice])
    fake_serving_endpoints = MagicMock()
    fake_serving_endpoints.query.return_value = fake_response
    fake_workspace_client = MagicMock(serving_endpoints=fake_serving_endpoints)
    monkeypatch.setattr(
        "datacontract.ai.databricks_pii_detector.WorkspaceClient",
        MagicMock(return_value=fake_workspace_client),
    )

    config = Config(databricks_pii_endpoint="pii-endpoint", databricks_profile="DEFAULT")
    detector = DatabricksPiiDetector(config)

    with pytest.raises(DataContractException, match="Could not classify columns"):
        detector.detect(["email"])
