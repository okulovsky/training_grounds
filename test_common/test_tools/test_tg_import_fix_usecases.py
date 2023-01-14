from tg.common.tools.tg_imports_fix import ImportData, ImportFixer
from unittest import TestCase
from pathlib import Path

class TgImportUseCasesTestCase(TestCase):
    def test_inside(self):
        imp = ImportData(
            Path('/repo/tg/grammar_ru/__init__.py'),
            0,
            'tg.grammar_ru.corpus',
            'x'
        )
        fixer = ImportFixer(
            Path('/repo/tg'),
            'tg'
        )
        imp = fixer.fix_absolute_import(imp)
        print(imp.from_module)

