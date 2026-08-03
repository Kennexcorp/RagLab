"""
Tests for dependency contracts that are not visible from our own imports.
"""


def test_torchvision_is_installed():
    """torchvision must stay a declared dependency even though no RAGLab module imports it.

    transformers imports torchvision unguarded at the top of its image-processor
    modules (e.g. models/detr/image_processing_detr.py) while declaring it only
    under its optional "vision" extra. Streamlit's module watcher walks sys.modules
    and triggers those lazy imports, so dropping torchvision crashes the GUI with
    ModuleNotFoundError even though nothing here references it.
    """
    import torchvision  # noqa: F401
