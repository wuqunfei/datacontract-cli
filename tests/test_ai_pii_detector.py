from datacontract.ai.annotate import mark_pii_columns
from datacontract.ai.pii_detector import PiiDetector
from datacontract.ai.rule_based_pii_detector import RuleBasedPiiDetector
from datacontract.imports.odcs_helper import create_odcs, create_property, create_schema_object


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
