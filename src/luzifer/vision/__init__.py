"""NDI still-frame vision: grab → caption → optional speech."""

from luzifer.vision.frames import frame_changed, jpeg_bytes
from luzifer.vision.ndi import NdiGrabber, list_ndi_sources
from luzifer.vision.observation import Observation, ObservationStore
from luzifer.vision.ollama import VisionOllama

__all__ = [
    "NdiGrabber",
    "Observation",
    "ObservationStore",
    "VisionOllama",
    "frame_changed",
    "jpeg_bytes",
    "list_ndi_sources",
]
