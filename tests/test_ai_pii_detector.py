from datacontract.ai.rule_based_pii_detector import RuleBasedPiiDetector


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
