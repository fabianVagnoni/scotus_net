#!/usr/bin/env python3
"""
SCOTUS AI Augmentation Module
============================

This module provides text augmentation capabilities for Supreme Court case data.
It includes augmentation for justice biographies, case descriptions, and case metadata.

Modules:
- justice_bios_augmentation: Augment justice biographies
- case_descriptions_augmentation: Augment case descriptions  
- main: Main augmentation pipeline orchestrator
"""

__version__ = "1.0.0"
__author__ = "SCOTUS AI Team"

from .augmenter import Augmenter, create_augmenter, augment_text_list
from .justice_bios_augmentation import create_augmented_bios
from .case_descriptions_augmentation import create_augmented_case_descriptions
from .main import run_augmentation_pipeline

__all__ = [
    'create_augmented_bios',
    'create_augmented_case_descriptions', 
    'run_augmentation_pipeline'
] 