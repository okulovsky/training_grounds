from yo_fluq_ds import Query
from enum import Enum


class Autonamer:
    def __init__(self, build_method, prefix = None, common_arguments = None):
        self.build_method = build_method
        self.prefix = prefix
        self.common_arguments = {} if common_arguments is None else common_arguments

    def _build_arguments(self, kwargs):
        grid = {}
        plain = {}
        for k,v in kwargs.items():
            if isinstance(v, list):
                grid[k]=v
            else:
                plain[k]=v
        result = Query.combinatorics.grid(**grid).to_list()
        for r in result:
            for k, v in plain.items():
                r[k] = v
        return result

    def build_tasks(self, **kwargs):
        calls = self._build_arguments(kwargs)
        result = []
        for call in calls:
            parts = []
            for key, value in call.items():
                prefix = ''.join([c[0].upper() for c in key.split('_')])
                v = ''
                if isinstance(value, int):
                    if value >= 1000000:
                        v = str(value // 1000000) + "M"
                    elif value >= 1000:
                        v = str(value // 1000) + "K"
                    else:
                        v = str(value)
                elif isinstance(value, list) or isinstance(value, tuple):
                    v = '-'.join(str(c) for c in value)
                elif isinstance(value, float):
                    v = str(value).replace('.', '')
                elif isinstance(value, Enum):
                    v = ''.join([c for c in value.name if c.isupper() or c.isdigit()])
                else:
                    v = str(value)
                parts.append(prefix + v)
            name_suffix = '-'.join(parts)
            for key, value in self.common_arguments.items():
                call[key] = value
            task = self.build_method(**call)
            if 'name' not in task.info:
                task.info['name'] = ''
            if self.prefix is not None:
                task.info['name'] = self.prefix+'-'+task.info['name']
            task.info['name'] += name_suffix
            result.append(task)
        return result
