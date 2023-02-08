from unittest import TestCase
from tg.common.tools.tg_imports_fix import *
from pprint import pprint
from pathlib import Path
from tg.common import Loc

class ImportFixTestCase(TestCase):
    def test_absolute_module_name_to_path(self):
        r = ImportFixer.absoulute_module_name_to_path(Loc.tg_path, 'tg', 'tg.common.ml.batched_training')
        self.assertEqual(Loc.tg_path/'common/ml/batched_training', r)

    def test_common_prefix(self):
        path1 = '/a/bf.d/we.asda/qwe.adsa'
        path2 = '/a/bf.d/fdasda/qwqw/xas/asd'
        c = ImportFixer.find_common_prefix(Path(path1), Path(path2))
        self.assertEqual('/a/bf.d', str(c))

    def test_module_form(self):
        path1 = '/a/b/c/d/e'
        path2 = '/a/b/x/y/z/p.py'
        c = ImportFixer.relative_import(Path(path2), Path(path1))
        self.assertEqual('...c.d.e', c)



    def test_parse(self):
        text = '''
from pandas import X
from tg.common import Loc
from ...common import Loc
from ...common.ml import batched_training as bt
import numpy as np

def something():
    pass
        '''
        res = ImportFixed.parse_imports(text)
        res = [r.__dict__ for r in res]
        self.assertListEqual(
            [{'content': 'X',
              'file': None,
              'from_module': 'pandas',
              'line_number': 'from pandas import X'},
             {'content': 'Loc',
              'file': None,
              'from_module': 'tg.common',
              'line_number': 'from tg.common import Loc'},
             {'content': 'Loc',
              'file': None,
              'from_module': '...common',
              'line_number': 'from ...common import Loc'},
             {'content': 'batched_training as bt',
              'file': None,
              'from_module': '...common.ml',
              'line_number': 'from ...common.ml import batched_training as bt'}
             ], res)
