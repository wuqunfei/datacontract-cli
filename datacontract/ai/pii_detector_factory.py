import importlib
import typing
from enum import Enum

from datacontract.ai.pii_detector import PiiDetector

if typing.TYPE_CHECKING:
    from datacontract.config import Config


class PiiDetectionMethod(str, Enum):
    rule = "rule"
    anthropic = "anthropic"
    databricks = "databricks"


class PiiDetectorFactory:
    def __init__(self) -> None:
        self._lazy_detectors: dict[str, tuple[str, str]] = {}

    def register_lazy_detector(self, name: str, module_path: str, class_name: str) -> None:
        self._lazy_detectors[name] = (module_path, class_name)

    def create(self, name: str, config: "Config | None" = None) -> PiiDetector:
        if name not in self._lazy_detectors:
            raise ValueError(f"The '{name}' PII detector is not supported.")
        module_path, class_name = self._lazy_detectors[name]
        module = importlib.import_module(module_path)
        detector_class = getattr(module, class_name)
        return detector_class(config)


pii_detector_factory = PiiDetectorFactory()
pii_detector_factory.register_lazy_detector(
    PiiDetectionMethod.rule, "datacontract.ai.rule_based_pii_detector", "RuleBasedPiiDetector"
)
pii_detector_factory.register_lazy_detector(
    PiiDetectionMethod.anthropic, "datacontract.ai.anthropic_pii_detector", "AnthropicPiiDetector"
)
pii_detector_factory.register_lazy_detector(
    PiiDetectionMethod.databricks, "datacontract.ai.databricks_pii_detector", "DatabricksPiiDetector"
)
