"""NVIDIA NIM provider package."""

from providers.defaults import NVIDIA_NIM_DEFAULT_BASE

from .client import NvidiaNimProvider
from .key_rotator import NimKeyRotator

__all__ = ["NVIDIA_NIM_DEFAULT_BASE", "NvidiaNimProvider", "NimKeyRotator"]
