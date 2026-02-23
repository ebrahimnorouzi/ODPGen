import re


def evaluate_structure_similarity(ontology_text: str) -> dict:
    class_count = len(re.findall(r"\ba\s+owl:Class\b", ontology_text))
    object_property_count = len(re.findall(r"\ba\s+owl:ObjectProperty\b", ontology_text))
    datatype_property_count = len(re.findall(r"\ba\s+owl:DatatypeProperty\b", ontology_text))
    return {
        "class_count": class_count,
        "object_property_count": object_property_count,
        "datatype_property_count": datatype_property_count,
        "similarity_mode": "structure_proxy_only",
    }
