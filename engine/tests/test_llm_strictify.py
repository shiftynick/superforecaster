"""Tests for the OpenAI-strict-mode schema normalization in the Codex client."""

from superforecaster.llm import _strictify_schema


def test_adds_additional_properties_false_to_objects() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "nested": {"type": "object", "properties": {"x": {"type": "string"}}},
            "items": {"type": "array", "items": {"type": "object", "properties": {}}},
            "plain": {"type": "number"},
        },
    }
    strict = _strictify_schema(schema)

    assert strict["additionalProperties"] is False
    props = strict["properties"]
    assert isinstance(props, dict)
    assert props["nested"]["additionalProperties"] is False
    assert props["items"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in props["plain"]


def test_does_not_mutate_input_or_override_existing() -> None:
    schema: dict[str, object] = {"type": "object", "additionalProperties": True}
    strict = _strictify_schema(schema)
    # Existing explicit value is preserved (caller said so deliberately)...
    assert strict["additionalProperties"] is True
    # ...and the original is untouched by the walk.
    assert schema == {"type": "object", "additionalProperties": True}
