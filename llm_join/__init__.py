try:
    import numpy as np
    import faiss  # noqa: F401
except ImportError as e:
    if "numpy._core" in str(e) or "numpy.core" in str(e):
        raise ImportError(
            "llm-join: numpy/faiss version mismatch detected.\n\n"
            "Fix with:\n"
            "  pip install 'faiss-cpu>=1.8' --force-reinstall\n\n"
            f"Original error: {e}"
        ) from None
    raise

from llm_join.join import fuzzy_join

__all__ = ["fuzzy_join"]
__version__ = "0.2.5"
