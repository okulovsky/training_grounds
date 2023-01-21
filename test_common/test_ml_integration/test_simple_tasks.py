from unittest import TestCase
from tg.common.ml.batched_training import sandbox as bts
from tg.common.ml.batched_training import factories as btf
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import InMemoryTrainingEnvironment
from sklearn.metrics import roc_auc_score

def get_bundle_and_task():
    bundle = bts.get_binary_classification_bundle()
    task = bts.SandboxTorchTask(
        [
            bts.get_feature_extractor(),
            bts.get_binary_label_extractor()
        ],
        (100,),
        roc_auc_score
    )
    task.settings.epoch_count=10
    return bundle, task


class PytorchTestCase(TestCase):
    def test_binary_classification(self):
        bundle, task = get_bundle_and_task()
        env = InMemoryTrainingEnvironment()
        task.run_with_environment(bundle, env)
        result = env.result
        self.assertGreater(result['metrics']['roc_auc_score_test'], 0.7)
        self.assertEqual(10, len(task.history))

    def test_multiclass_classification(self):
        bundle = bts.get_multilabel_classification_bundle()
        task = bts.SandboxTorchTask(
            [
                bts.get_feature_extractor(),
                bts.get_multilabel_extractor()
            ],
            (30,),
            bt.MulticlassMetrics(True, True, [1])
        )
        task.settings.epoch_count = 10
        task.optimizer_ctor = btf.CtorAdapter('torch.optim:Adam', ('params',), lr = 0.1)
        env = InMemoryTrainingEnvironment()
        task.run_with_environment(bundle, env)
        result = env.result
        self.assertGreater(result['metrics']['recall_at_1_test'], 0.7)


    def test_get_batch(self):
        bundle, task = get_bundle_and_task()
        batch, data = task.generate_sample_batch_and_temp_data(bundle, 0)
        self.assertSetEqual({'features', 'label'}, set(batch.bundle.data_frames))

    def test_get_metrics(self):
        bundle, task = get_bundle_and_task()
        self.assertSetEqual(
            {'roc_auc_score_train', 'roc_auc_score_test', 'roc_auc_score_display', 'loss', 'iteration'},
            set(task.get_metric_names())
        )

    def test_minibatches(self):
        bundle, task = get_bundle_and_task()
        task.settings.batch_size = 100
        task.settings.mini_batch_size = 20
        task.settings.mini_epoch_count = 3
        env = InMemoryTrainingEnvironment()
        task.run_with_environment(bundle, env)
        print(env.result['metrics']['roc_auc_score_test'])
        # for s in env.message_buffer:
        #    print(s)
