from typing import *
from ... import batched_training as bt
from .conventions import Conventions
from .torch_model_handler import TorchModelHandler
from .networks.basics import CtorAdapter

class AssemblyPoint:
    def create_network_factory(self):
        raise NotImplementedError()

    def create_extractor(self):
        raise NotImplementedError()


def _initialization_bridge(task, data):
    return task.initialize_task(data)


class TorchTrainingTask(bt.BatchedTrainingTask):
    def __init__(self):
        splitter = bt.PredefinedSplitter(
                Conventions.SplitColumnName,
                [Conventions.TestName, Conventions.DisplayName],
                [Conventions.TrainName, Conventions.DisplayName]
            )
        super(TorchTrainingTask, self).__init__(
            settings = bt.TrainingSettings(),
            splitter = splitter,
            metric_pool= bt.MetricPool(),
            late_initialization=_initialization_bridge
        )
        self.optimizer_ctor = CtorAdapter('torch.optim:SGD', ('params',), lr = 0.1)
        self.loss_ctor = CtorAdapter('torch.nn:MSELoss')

    def initialize_task(self, data):
        raise NotImplementedError()

    def setup_batcher(self, ibundle, extractors, index_frame_name='index', stratify_by_column = None):
        strategy = None
        if stratify_by_column is not None:
            df = ibundle.bundle[index_frame_name]
            df[Conventions.PriorityColumnName] = bt.PriorityRandomBatcherStrategy.make_priorities_for_even_representation(df, stratify_by_column)
            strategy = bt.PriorityRandomBatcherStrategy(Conventions.PriorityColumnName)
        self.batcher = bt.Batcher(extractors, strategy)


    def setup_model(self, network_factory):
        self.model_handler = TorchModelHandler(network_factory, self.optimizer_ctor, self.loss_ctor)



