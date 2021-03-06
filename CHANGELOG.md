# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added `edexplore` for dataset exploration with streamlit: `edexplore -b <config.yaml>`
- Added Late Loading! You can now return functions in your examples, which will only be evaluated at the end of you data processing pipeline, allowing you to stack many filter operations on top of each other.
- Added MetaView Dataset, which allows to store views on a base dataset without the need to recalculate the labels everytime.
- `TFBaseEvaluator` now parses config file for `fcond` flag to filter checkpoints, e.g.`edflow -e xxx --fcond "lambda c: any([str(n) in c for n in [240000, 320000]])"` will only evaluate checkpoint 240k and 320k
- Added MetaDataset for easy Dataloading
- Added CelebA dataset.
- Added CIFAR10 dataset.
- Environment variable EDFLOW\_GIT enables git integration.
- Minimal logger now supports list of handlers.
- pdb support. Use `import edflow.fpdb as pdb; pdb.set_trace()` instead of
  `import pdb; pdb.set_trace()`.
- TFMultiStageTrainer and TFMultiStageModel
    - adds interface for model and trainer to derive from to support multiple stages. For example pretraining and training.
    - example can be found in `examples/multistage_trainer`
- `edlist` shows process group id (`pgid`), see FAQ.
- method `probs_to_mu_sigma` in `tf_nn.py` to estimate spatial mean and covariance across multiple channels.
- CHANGELOG.md to document notable changes.

### Changed
- When setting the `DatasetMixin` attribute `append_labels = True` the labels are not added to the example directly but behind the key `labels_`.
- Changed tiling background color to white
- Changed interface of `edflow.data.dataset.RandomlyJoinedDataset` to improve it.
- Removed colons from log directory names and replaced them by hyphens.
- `LambdaCheckpointHook` uses global step and doesn't save on first step.
- Switched opencv2 functions with manual ones to get rid of the dependency.
- `edeval` now allows for differnet callback interface via the config. Callbacks are now entered as `dicts`, which allows to also pass keyword arguments to the callbacks from the config.
- `make_batches` now produces deep batches of examples. See documentation of `deep_lod2dol` or the section "Datasets and Batching" in the documention.
- `make_var` was broken for period variables because subcommands lacked `**kwargs` in definition. This is fixed now.

### Removed
- It is no longer possible to pass callbacks as list via the config
