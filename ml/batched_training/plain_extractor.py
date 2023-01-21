from typing import *

import pandas as pd
import copy

from enum import Enum

from .extractors import Extractor, DataBundle, IndexedDataBundle


class _JoinType(Enum):
    NullIndex = 0
    Index = 1
    Join = 2


class _JoinDescription:
    def __init__(self, frame: Optional[str], join_type: _JoinType):
        self.frame = frame
        self.join_type = join_type
        self.keep_columns = None  # type: Optional[List[str]]

    def _set_columns(self, columns: Union[str, List[str], None]):
        if columns is None:
            self.keep_columns = None
        elif isinstance(columns, list):
            self.keep_columns = columns
        else:
            self.keep_columns = [columns]


class PlainExractorBuilder:
    def __init__(self, name: str):
        self.name = name
        self.joins = []  # type: List[_JoinDescription]

    def index(self, frame_name=None) -> 'PlainExractorBuilder':
        if len(self.joins) != 0:
            raise ValueError('`index` method may appear only once, at the very beginning')
        if frame_name is None:
            self.joins.append(_JoinDescription(None, _JoinType.NullIndex))
        else:
            self.joins.append(_JoinDescription(frame_name, _JoinType.Index))
        return self

    def join(self, frame_name: str, on_columns=Union[str, List[str]]) -> 'PlainExractorBuilder':
        if len(self.joins) == 0:
            self.joins.append(_JoinDescription(None, _JoinType.NullIndex))
        self.joins[-1]._set_columns(on_columns)
        self.joins.append(_JoinDescription(frame_name, _JoinType.Join))
        return self

    def apply(self,
              transformer: Optional = None,
              take_columns: Union[str, List[str], None] = None,
              raise_if_rows_are_missing: bool = True,
              raise_if_nulls_detected: bool = True,
              coalesce_nulls: Optional = None,
              drop_columns: Union[str, List[str], None] = None
              ) -> 'PlainExtractor':
        if len(self.joins) == 0:
            self.joins.append(_JoinDescription(None, _JoinType.NullIndex))
        self.joins[-1]._set_columns(take_columns)
        return PlainExtractor(
            self.name,
            self.joins,
            transformer,
            raise_if_rows_are_missing,
            raise_if_nulls_detected,
            coalesce_nulls,
            drop_columns
        )


class PlainExtractor(Extractor):
    def __init__(self,
                 name: str,
                 joins: List[_JoinDescription],
                 transformer: Optional,
                 raise_if_rows_are_missing: bool,
                 raise_if_nulls_detected: bool,
                 coalesce_nulls: Optional = None,
                 drop_columns: Optional = None
                 ):
        self.name = name
        self.joins = joins
        self.transformer = copy.deepcopy(transformer)
        self.raise_if_rows_are_missing = raise_if_rows_are_missing
        self.raise_if_nulls_detected = raise_if_nulls_detected
        self.coalesce_nulls = coalesce_nulls
        self.drop_columns = drop_columns

    @staticmethod
    def build(name: str):
        return PlainExractorBuilder(name)

    def _build_frame(self, ibundle: IndexedDataBundle):
        if ibundle.index_frame.index.duplicated().any():
            raise ValueError('`PlainExtractor` does not support extraction if samples are duplicated')

        current = None
        first_join = self.joins[0]
        if first_join.join_type == _JoinType.NullIndex:
            current = ibundle.index_frame
        elif first_join.join_type == _JoinType.Index:
            current = ibundle.bundle[first_join.frame].loc[ibundle.index_frame.index]
        else:
            raise ValueError('First join must be NullIndex or Index')
        if first_join.keep_columns is not None:
            current = current[first_join.keep_columns]

        for join_index, join in enumerate(self.joins[0:]):
            if join_index == 0:
                continue
            if join.join_type != _JoinType.Join:
                raise ValueError('All joins except first must be Join')
            frame = ibundle.bundle[join.frame]
            if join.keep_columns is not None:
                missing_columns = [c for c in join.keep_columns if c not in frame.columns]
                if len(missing_columns)>0:
                    raise ValueError(f'The following columns are missing: {missing_columns}')
                frame = frame[join.keep_columns]

            if self.raise_if_rows_are_missing:
                how = 'inner'
            else:
                how = 'left'

            join_columns = self.joins[join_index - 1].keep_columns
            current = current.merge(frame, left_on=join_columns, right_index=True, how=how).drop(join_columns, axis=1)
            if current.shape[0] < ibundle.index_frame.shape[0]:
                raise ValueError(f'Error in extractor {self.name}: when merging with {join.frame}, less rows are produced, {current.shape[0]} instead {ibundle.index_frame.shape[0]}. Are some rows missing?')
            elif current.shape[0] > ibundle.index_frame.shape[0]:
                raise ValueError(f'Error in extractor {self.name}: When merging with {join.frame}, more rows are produced, {current.shape[0]} instead {ibundle.index_frame.shape[0]}. Is index non-unique?)')
            elif not (current.index == ibundle.index_frame.index).all():
                current = current.loc[ibundle.index_frame.index]
                if not (current.index == ibundle.index_frame.index).all():
                    raise ValueError(f'Error in extractor {self.name}: something wrong happened when merging with {join.frame}: same amount of rows, but indices are different')
        if self.drop_columns is not None:
            current = current.drop(self.drop_columns, axis=1)
        return current

    def fit(self, ibundle: IndexedDataBundle):
        if self.transformer is not None:
            frame = self._build_frame(ibundle)
            self.transformer.fit(frame)

    def extract(self, ibundle: IndexedDataBundle) -> pd.DataFrame:
        frame = self._build_frame(ibundle)
        if self.transformer is not None:
            frame = self.transformer.transform(frame)
        if self.raise_if_nulls_detected:
            if frame.isnull().any().any():
                null_columns = frame.isnull().any(axis=0)
                null_columns = list(null_columns.loc[null_columns].index)
                raise ValueError(f'Error in extractor {self.name}: nulls are detected in the output in columns\n{null_columns}')
        elif self.coalesce_nulls is not None:
            frame = frame.fillna(value = self.coalesce_nulls)
        return frame

    def get_name(self):
        return self.name
