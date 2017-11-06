# -*- coding: utf-8 -*-

__all__ = ["TestCase"]

import os
import sys
import unittest

base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)
from utils.tools import *

try:
    from utils.model import KerasModels
    from utils.treetools import TreeExtender
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
        self.assertEqual(model.ismethod(), True)

    @if_treetool
    def test_treetool(self):
        tool = TreeExtender()
        self.assertEqual(tool.ismethod(), True)

    def test_flattenlist(self):
        a = [[1., 2.], [3.]]
        self.assertEqual(flattenList(a), [1., 2., 3.])

    def test_matchingItem(self):
        string = 'foo'
        regex1 = 'f'
        regex2 = 'a'
        self.assertEqual(matchingItem(regex1, string), 'f')
        self.assertEqual(matchingItem(regex2, string), None)
