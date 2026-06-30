"""Unit tests for the request schema (input validation)."""
import pytest
from pydantic import ValidationError

from api.main import IrisFeatures


def test_valid_payload_is_accepted():
    f = IrisFeatures(
        sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
    )
    assert f.petal_length == 1.4


def test_integer_inputs_are_coerced_to_float():
    f = IrisFeatures(sepal_length=5, sepal_width=3, petal_length=1, petal_width=0)
    assert isinstance(f.sepal_length, float)
    assert f.sepal_length == 5.0


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"sepal_length": "not-a-number", "sepal_width": 3.5,
         "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4},
    ],
)
def test_invalid_payload_is_rejected(bad_payload):
    with pytest.raises(ValidationError):
        IrisFeatures(**bad_payload)
