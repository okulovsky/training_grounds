from typing import *

import os
import subprocess
import uuid

from pathlib import Path


def _get_git_user():
    try:
        output = subprocess.run(
            ['git', 'config', '--list'], capture_output=True, text=True).stdout
        username = output.split('email=')[1].split('@')[0]
        return username
    except:
        return 'default'


def _norm_relative_path(p):
    path = Path(__file__).parent.joinpath(p)
    path = os.path.normpath(path)
    return Path(path)


class LocationsClass:
    def __init__(self):
        self.root_path = _norm_relative_path('../../../')
        self.tg_path = _norm_relative_path('../../')
        self.tg_common_path = _norm_relative_path('../')
        self.data_cache_path = self.root_path.joinpath('data-cache')
        self.temp_path = self.root_path.joinpath('temp')
        self.tg_name = str(self.tg_path.relative_to(self.root_path))
        self.git_username = _get_git_user()

        import_path = type(self).__module__
        suffix = '.common._common.locations'
        if not import_path.endswith(suffix):
            self.tg_common_path = None
        else:
            self.tg_import_path = import_path.replace(suffix,'')

    def get_default_temp_path(self, prefix: str, location: Optional[Union[str, Path]] = None) -> Path:
        if location is None:
            return self.temp_path / prefix / str(uuid.uuid4())
        return Path(location)


Loc = LocationsClass()
