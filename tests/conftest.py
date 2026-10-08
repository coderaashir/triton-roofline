import os

import pytest


@pytest.fixture
def device():
    """Where kernel tests run: the GPU if there is one, else Triton's CPU interpreter."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("no GPU; set TRITON_INTERPRET=1 to run kernels on CPU")
    return "cpu"
