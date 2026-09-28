from typing import List

from open_data_contract_standard.model import OpenDataContractStandard, SchemaProperty

from datacontract.ai.pii_detector import PiiDetector
from datacontract.imports.odcs_helper import walk_properties


def mark_pii_columns(odcs: OpenDataContractStandard, detector: PiiDetector) -> None:
    """Mark likely-PII columns with classification="PII" and criticalDataElement=True.

    Never overwrites a property that already has a classification, and never calls
    the detector for a table where every column is already classified.
    """
    for schema_obj in odcs.schema_ or []:
        properties: List[SchemaProperty] = [prop for _, prop in walk_properties(schema_obj.properties)]
        candidates = [prop for prop in properties if prop.classification is None]
        if not candidates:
            continue
        results = detector.detect([prop.name for prop in candidates])
        for prop in candidates:
            if results.get(prop.name):
                prop.classification = "PII"
                prop.criticalDataElement = True
