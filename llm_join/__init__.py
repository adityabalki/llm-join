try:
    import numpy as np
    import faiss  # noqa: F401
except ImportError as e:
    if "numpy._core" in str(e) or "numpy.core" in str(e):
        raise ImportError(
            "llm-join: numpy/faiss version mismatch detected.\n\n"
            "faiss-cpu>=1.14 requires numpy>=2, which breaks numpy 1.x environments.\n\n"
            "Fix with:\n"
            "  pip install \"faiss-cpu>=1.8,<1.14\" \"numpy>=1.23,<2\" --force-reinstall\n\n"
            "On Databricks, run the above as %pip install ... then restart Python:\n"
            "  dbutils.library.restartPython()\n\n"
            f"Original error: {e}"
        ) from None
    raise

from llm_join.join import fuzzy_join

__all__ = ["fuzzy_join"]
__version__ = "0.4.0"
