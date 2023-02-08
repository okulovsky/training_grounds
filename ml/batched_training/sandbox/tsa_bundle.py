from pathlib import Path
from ...._common import DataBundle


def get_tsa_bundle():
    return DataBundle.load(Path(__file__).parent/'tsa-test.zip')
