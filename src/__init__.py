"""
Trident - Synthetic Aperture Radar Maritime Vessel Detection System

A production-grade maritime surveillance system for detecting ships in SAR imagery
using Oriented Bounding Boxes (OBB) and speckle noise filtering.
"""

__version__ = "1.0.0"
__author__ = "Trident Team"

from .preprocessing import SARPreprocessor, apply_lee_filter
from .data_manager import SSDDDataManager
