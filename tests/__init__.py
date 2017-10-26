# -*- coding: utf-8 -*-

__all__ = ["TestCase"]

import os
import sys
import unittest

base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)

try:
    from model import KerasModels
    from treetools import TreeExtender
    HAS_MODEL = True
    HAS_TREETOOLS = True
except ImportError:
    KerasModels = None
    TreeExtender = None
    HAS_MODEL = False
    HAS_TREETOOLS = False


def if_kerasmodel(model):
    return model if HAS_MODEL else (lambda self: None)


def if_treetool(tool):
    return tool if HAS_TREETOOLS else (lambda self: None)


class TestCase(unittest.TestCase):
    @if_kerasmodel
    def test_kerasmodel(self):
        model = KerasModels.example_model()
        self.assertEqual(self.model.ismethod(), True)

    @if_treetool
    def test_treetool(self):
        tool = TreeExtender()
        self.assertEqual(self.tool.ismethod(), True)
