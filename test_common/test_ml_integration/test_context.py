from tg.common.ml.batched_training import context as btc
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training.sandbox import AlternativeTrainingTask
from unittest import TestCase
from pathlib import Path


class AlternativeTestCase(TestCase):
    def run_test(self, reduction_type, network_type):
        task = AlternativeTrainingTask()
        task.settings.epoch_count = 1
        task.settings.batch_size = 20000
        task.settings.mini_epoch_count = 10
        task.optimizer_ctor.type='torch.optim:Adam'
        task.context.reduction_type = reduction_type
        task.context.dim_3_network_factory.network_type = network_type
        result = task.run(Path(__file__).parent/'tsa-test.zip')
        self.assertGreater(result['metrics']['roc_auc_score_test'], 0.7)



    def test_pivot(self):
        self.run_test(btc.ReductionType.Pivot, None)

    def test_lstm(self):
        self.run_test(btc.ReductionType.Dim3, btc.Dim3NetworkType.LSTM)

    def test_lstm_with_attention(self):
        self.run_test(btc.ReductionType.Dim3, btc.Dim3NetworkType.SelfAttentionAndLSTM)

    def test_alon_attention(self):
        self.run_test(btc.ReductionType.Dim3, btc.Dim3NetworkType.AlonAttention)

    def test_lstm_folded(self):
        self.run_test(btc.ReductionType.Dim3Folded, btc.Dim3NetworkType.LSTM)
