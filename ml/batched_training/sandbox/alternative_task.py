from ... import batched_training as bt
from ... import dft
from .. import factories as btf
from .. import context as btc
from .plain_context_builder import PlainContextBuilder
from sklearn.metrics import roc_auc_score

class AlternativeTrainingTask(btf.TorchTrainingTask):
    def __init__(self):
        super(AlternativeTrainingTask, self).__init__()
        self.settings.mini_batch_size = 200
        self.settings.mini_epoch_count = 1
        core_extractor = bt.PlainExtractor.build('features').index().join('pymorphy', 'another_word_id').apply(
            transformer=dft.DataFrameTransformerFactory.default_factory(),
            drop_columns = ['normal_form']
        )

        self.context = btc.ContextualAssemblyPoint(
            'features',
            PlainContextBuilder(
                include_zero_offset=True,
                left_to_right_contexts_proportion=0.5
            ),
            core_extractor,
        )

        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)

        self.tail_size = 50

    def initialize_task(self, idb: bt.IndexedDataBundle):
        label_extractor = (
            bt.PlainExtractor
            .build(btf.Conventions.LabelFrame)
            .index()
            .apply(
                take_columns='label',
                transformer=None
            )
        )
        extractors = [
            self.context.create_extractor(),
            label_extractor
        ]

        self.setup_batcher(idb, extractors)

        head_factory = self.context.create_network_factory()
        factory = btf.FeedForwardNetwork.Factory(
            head_factory,
            btf.Factories.Factory(btf.Perceptron, output_size=self.tail_size),
            btf.Factories.Factory(btf.Perceptron, output_size=1)
        )
        self.setup_model(factory, True)