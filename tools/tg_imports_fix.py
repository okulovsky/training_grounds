import re
from pathlib import Path
import copy
from yo_fluq_ds import FileIO

import_re = re.compile('from ([^ ]+) import (.*)')

class ImportData:
    def __init__(self, file:Path, line_number:int, from_module:str, content:str):
        self.from_module = from_module
        self.content = content
        self.file = file
        self.line_number = line_number


class ImportFixer:
    def __init__(self, tg_root: Path, tg_name = 'tg'):
        self.tg_root = tg_root
        self.tg_name = tg_name

    def fix_absolute_import(self, imp: ImportData):
        if not imp.from_module.startswith(self.tg_name+'.'):
            return None
        due_path = ImportFixer.absoulute_module_name_to_path(self.tg_root, self.tg_name, imp.from_module)
        due_module = ImportFixer.relative_import(imp.file, due_path)
        imp = copy.copy(imp)
        imp.from_module = due_module
        return imp

    def apply(self, imp: ImportData):
        print(imp.__dict__)
        lines = FileIO.read_text(imp.file).split('\n')
        ln = f'from {imp.from_module} import {imp.content}'
        lines[imp.line_number] = ln
        FileIO.write_text('\n'.join(lines), imp.file)



    @staticmethod
    def relative_import(from_file: Path, to_module: Path):
        common = ImportFixer.find_common_prefix(from_file, to_module)
        dots = '.'
        from_rel = str(from_file.relative_to(common))
        for i in from_rel:
            if i=='/':
                dots+='.'
        remainder = to_module.relative_to(common)
        return dots+str(remainder).replace('/','.')




    @staticmethod
    def absoulute_module_name_to_path(tg_root, tg_name, module):
        path = module[len(tg_name)+1:]
        path = path.replace('.', '/')
        path = tg_root / path
        return path


    @staticmethod
    def find_common_prefix(path1: Path, path2: Path):
        while True:
            if str(path2).startswith(str(path1)):
                return path1
            path1 = path1.parent

    @staticmethod
    def parse_imports(text, file_name = None):
        results = []
        lines = text.split('\n')
        for line_number, line in enumerate(lines):
            match = import_re.match(line)
            if match is not None:
                results.append(ImportData(file_name, line_number, match.group(1), match.group(2)))
        return results

    @staticmethod
    def parse_imports_from_file(file_name):
        return ImportFixer.parse_imports(FileIO.read_text(file_name), file_name)




