import os
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.checkpoint_hooks.lambda_checkpoint_hook import LambdaCheckpointHook
from edflow.hooks.logging_hooks.minimal_logging_hook import LoggingHook
from edflow.hooks.util_hooks import IntervalHook, ExpandHook
from edflow.eval.pipeline import TemplateEvalHook
from edflow.project_manager import ProjectManager
from edflow.util import retrieve, set_default
from edflow.main import get_obj_from_str


class TemplateIterator(PyHookedModelIterator):
    """A specialization of PyHookedModelIterator which adds reasonable default
    behaviour. Subclasses should implement `save`, `restore` and `step_op`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # wrap save and restore into a LambdaCheckpointHook
        self.ckpthook = LambdaCheckpointHook(
            root_path=ProjectManager.checkpoints,
            global_step_getter=self.get_global_step,
            global_step_setter=self.set_global_step,
            save=self.save,
            restore=self.restore,
            interval=set_default(self.config, "ckpt_freq", None),
        )
        # write checkpoints after epoch or when interrupted during training
        if not self.config.get("test_mode", False):
            self.hooks.append(self.ckpthook)

        ## hooks - disabled unless -t is specified

        # execute train ops
        self._train_ops = set_default(self.config, "train_ops",
                                      ["train/train_op"])
        train_hook = ExpandHook(paths=self._train_ops, interval=1)
        self.hooks.append(train_hook)

        # log train/step_ops/log_ops in increasing intervals
        self._log_ops = set_default(self.config, "log_ops",
                                    ["train/log_op", "validation/log_op"])
        self.loghook = LoggingHook(paths=self._log_ops,
                                   root_path=ProjectManager.train, interval=1)
        self.ihook = IntervalHook([self.loghook],
                                  interval=set_default(self.config,
                                                       "start_log_freq", 1),
                                  modify_each=1,
                                  max_interval=set_default(self.config,
                                                           "log_freq", 1000),
                                  get_step=self.get_global_step,)
        self.hooks.append(self.ihook)

        # setup logging integrations
        if not self.config.get("test_mode", False):
            default_wandb_logging = {"active": False,
                             "handlers": ["scalars", "images"]}
            wandb_logging = set_default(self.config, "integrations/wandb",
                                        default_wandb_logging)
            if wandb_logging["active"]:
                import wandb
                from edflow.hooks.logging_hooks.wandb_handler import (
                    log_wandb, log_wandb_images)

                os.environ["WANDB_RESUME"] = "allow"
                os.environ["WANDB_RUN_ID"] = ProjectManager.root.replace("/", "-")
                wandb.init(name=ProjectManager.root, config=self.config)

                handlers = set_default(self.config,
                                       "integrations/wandb/handlers",
                                       default_wandb_logging["handlers"])
                if "scalars" in handlers:
                    self.loghook.handlers["scalars"].append(log_wandb)
                if "images" in handlers:
                    self.loghook.handlers["images"].append(log_wandb_images)

            default_tensorboardX_logging = {"active": False,
                                    "handlers": ["scalars", "images"]}
            tensorboardX_logging = set_default(self.config,
                                               "integrations/tensorboardX",
                                               default_tensorboardX_logging)
            if tensorboardX_logging["active"]:
                from tensorboardX import SummaryWriter
                from edflow.hooks.logging_hooks.tensorboardX_handler import (
                    log_tensorboard_config,
                    log_tensorboard_scalars,
                    log_tensorboard_images,
                )

                self.tensorboardX_writer = SummaryWriter(ProjectManager.root)
                log_tensorboard_config(self.tensorboardX_writer, self.config,
                                       self.get_global_step())
                handlers = set_default(self.config,
                                       "integrations/tensorboardX/handlers",
                                       default_tensorboardX_logging["handlers"])
                if "scalars" in handlers:
                    self.loghook.handlers["scalars"].append(
                        lambda *args, **kwargs: log_tensorboard_scalars(
                            self.tensorboardX_writer, *args, **kwargs
                        )
                    )
                if "images" in handlers:
                    self.loghook.handlers["images"].append(
                        lambda *args, **kwargs: log_tensorboard_images(
                            self.tensorboardX_writer, *args, **kwargs
                        )
                    )

        ## epoch hooks

        # evaluate validation/step_ops/eval_op after each epoch
        self._eval_op = set_default(self.config, "eval_hook/eval_op",
                                    "validation/eval_op")
        label_key = set_default(self.config, "eval_hook/label_key",
                                "validation/eval_op/labels")
        self._eval_callbacks = set_default(self.config,
                                           "eval_hook/eval_callbacks", dict())
        if not isinstance(self._eval_callbacks, dict):
            self._eval_callbacks = {"cb": self._eval_callbacks}
        for k in self._eval_callbacks:
            self._eval_callbacks[k] = get_obj_from_str(self._eval_callbacks[k])
        callback_handler = None
        if not self.config.get("test_mode", False):
            callback_handler = lambda results, paths: self.loghook(
                results=results,
                step=self.get_global_step(),
                paths=paths,
            )
        self.evalhook = TemplateEvalHook(
            dataset=self.dataset,
            step_getter=self.get_global_step,
            keypath=self._eval_op,
            config=self.config,
            callbacks=self._eval_callbacks,
            labels_key=label_key,
            callback_handler=callback_handler,
        )
        self.epoch_hooks.append(self.evalhook)

    def initialize(self, checkpoint_path=None):
        if checkpoint_path is not None:
            self.ckpthook(checkpoint_path)

    def step_ops(self):
        return self.step_op

    def save(self, checkpoint_path):
        """Save state to checkpoint path."""
        raise NotImplemented()

    def restore(self, checkpoint_path):
        """Restore state from checkpoint path."""
        raise NotImplemented()

    def step_op(self, model, **kwargs):
        """Actual step logic. By default, a dictionary with keys 'train_op',
        'log_op', 'eval_op' and callable values is expected. 'train_op' should
        update the model's state as a side-effect, 'log_op' will be logged to
        the project's train folder. It should be a dictionary with keys
        'images' and 'scalars'. Images are written as png's, scalars are
        written to the log file and stdout. Outputs of 'eval_op' are written
        into the project's eval folder to be evaluated with `edeval`."""
        raise NotImplemented()
