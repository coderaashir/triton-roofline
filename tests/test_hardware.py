import pytest

from troof.hardware import T4


def test_t4_ridge_points():
    # 8.1e12 FLOP/s / 320e9 B/s ≈ 25 FLOPs per byte
    assert T4.ridge_point("fp32") == pytest.approx(25.3, abs=0.1)
    # tensor cores push the ridge much further right
    assert T4.ridge_point("fp16") == pytest.approx(203.1, abs=0.1)


def test_unknown_dtype_raises():
    with pytest.raises(KeyError):
        T4.ridge_point("fp8")
