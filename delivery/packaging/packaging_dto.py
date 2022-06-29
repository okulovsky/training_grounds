from typing import *

import re

from pathlib import Path
from yo_fluq_ds import Query, fluq

from ..._common import Loc


class DependenciesList:
    def __init__(self, name: str, dependencies: List[str]):
        self.name = name
        self.dependencies = dependencies


def get_default_dependencies() -> List[DependenciesList]:
    dependencies = Query.file.text(Path(__file__).parent / 'default_requirements.txt').to_list()
    buffer = [[]]
    for s in dependencies:
        if s == '':
            buffer.append([])
        else:
            buffer[-1].append(s)
    deps = Query.en(buffer).with_indices().select(lambda z: DependenciesList(f'core{z.key}', z.value)).to_list()
    return deps


class PackagingTask:
    """Class describing a Python package name"""

    def __init__(self,
                 name: str,
                 version: str,
                 payload: Dict[str, Any] = None,
                 additional_dependencies: Optional[List[DependenciesList]] = None,
                 silent=False
                 ):
        """

        Args:
            name: the name of the package. In TG, this is usually the project name + the milestone version
            version: the version. The packages with the same name but different versions cannot be installed at the same time
        """
        if not re.match('^[a-zA-Z0-9_-]+$', name):
            raise ValueError(f'Incorrect name: {name}')
        self.name = name
        self.version = version
        if payload is None:
            payload = {}
        self.payload = payload
        self.dependencies = get_default_dependencies()
        if additional_dependencies is not None:
            for lst in additional_dependencies:
                self.dependencies.append(lst)
        self.silent = silent


class PackageInfo:
    """
    Description of the created package
    """

    def __init__(self, task: PackagingTask, module_name: str, path: Path, properties: Any = None):
        """

        Args:
            module_name: the name of the root module in the package
            path: path to the file where the module is located
        """
        self.task = task
        self.module_name = module_name
        self.path = path
        self.properties = properties


class ContaineringTask:
    def __init__(self,
                 packaging_task: PackagingTask,
                 entry_file_name: str,
                 entry_file_template: str,
                 dockerfile_template: str,
                 image_name: str,
                 image_tag: str
                 ):
        self.packaging_task = packaging_task
        self.entry_file_name = entry_file_name
        self.entry_file_template = entry_file_template
        self.image_name = image_name
        self.image_tag = image_tag
        self.dockerfile_template = dockerfile_template
