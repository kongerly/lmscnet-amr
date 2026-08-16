# Experiment Configurations

This directory stores reproducible experiment configurations. Each configuration must explicitly record the experiment identifier, dataset version, split indices, run seed, model parameters, optimizer settings, and output path.

The RadioML 2016.10A preprocessing protocol is frozen to `per_sample_max_abs`. The historical test has already been consumed and is permanently locked; the controlled revision authorizes train/validation use only.

Dataset conversion contracts under `data/` are not experiment configurations. They freeze verified source identities, non-executable storage layouts, canonical sample identities, and artifact-publication rules before any converted data is written. Generated datasets and manifests remain outside the repository.
Dataset split contracts under `data/` freeze source bindings, deterministic stratification and assignment, leakage gates, and test isolation. A validated design contract is not a generated split and does not authorize test access.
Near-duplicate audit contracts under `data/` freeze signal representations, candidate-recall evidence, threshold calibration, pair review, and fail-closed audit publication. The approved benchmark split records the unresolved global near-duplicate limitation instead of treating the production audit as a generation prerequisite.
The near-duplicate fixture contract under `data/` binds a deterministic synthetic calibration and bounded exhaustive reference check. Fixture success is implementation evidence only and does not authorize a production audit or split.

Experiment configurations under `experiments/` are executable research contracts. `cnn2_radioml_2016_10a.yml` is the publication-candidate CNN2 configuration. Files containing `smoke` are bounded infrastructure checks and their metrics are not publication results.
`cldnn_radioml_2016_10a.yml` and `resnet1d_radioml_2016_10a.yml` provide train/validation-only starting configurations for the two additional baselines. They are executable starting points, not evidence that validation tuning has already selected their hyperparameters.
`cnn2_radioml_2016_10a_sweep.yml` fixes the validation-only learning-rate and dropout grid, sequential execution order, resume policy, and deterministic best-run tie-breaks. It never authorizes test access.
`na_lmscnet_wo_snr_auxiliary_radioml_2016_10a_selected.yml` is the seed-13 ablation screen. It inherits the selected NA-LMSCNet protocol and changes only the SNR auxiliary branch to a learned constant fusion embedding; test access remains forbidden.
`na_lmscnet_wo_multi_scale_radioml_2016_10a_selected.yml` and `na_lmscnet_fixed_average_radioml_2016_10a_selected.yml` are the module-7 seed-13 core screens. Both retain the frozen SNR auxiliary loss and change only the scale branches or fusion rule; their matching smoke configs are bounded infrastructure checks, and test access remains forbidden.

The final S0/S1/S2 family is defined by `lmscnet_s0_k{3,7,15}`, `lmscnet_s0_wide`, `lmscnet_s1`, and `lmscnet_s2` selected/smoke configurations. They share the same max-abs input protocol, augmentation, optimizer, budget, early stopping, and checkpoint rule. None contains an SNR head, SNR loss, SNR conditioning, or constant embedding. `lmscnet_s0_wide` is pre-registered as kernel `7`, expansion `1.42`, to match S2 MACs without selecting its width from validation outcomes.

The CNN2/CLDNN/ResNet1D selected files now match the completed max-abs seed-13 sweeps: CNN2 uses learning rate `3e-4` and dropout `0`; CLDNN and ResNet1D use learning rate `1e-3` and dropout `0`. `run_final_validation_family.py` completed all final-family and retuned baseline runs without test access.

The extended comparison adds `resnet1d_macs`, `mobilenetv2_1d`, `mcldnn`, and `se_msfn_1d` publication, sweep, and bounded-smoke configurations. `resnet1d_macs` is a pre-training MAC-matched control; `mobilenetv2_1d` is a parameter-matched MobileNetV2-style baseline; `mcldnn` is the strong temporal baseline adapted from the fixed MIT SigDA source; `se_msfn_1d` is the source-informed nearest multi-scale attention/fusion baseline. All use the same 2x2 seed-13 tuning grid before five-seed validation and keep test access forbidden.

The RadioML 2018.01A validation-only replication uses the four `*_radioml_2018_01a_selected.yml` configurations. They bind assignment `db3854fb698cd0b66a5ae67f1286535b06d890264ee3141895adafea0371fc01`, preserve the frozen 2016.10A training protocol, change only the dataset identity, class count, and required 1024-sample input compatibility, and run seeds 13, 37, and 73 without a validation sweep.

The controlled revision protocols are recorded in `revision/phase_r0.yml`, `revision/phase_r1.yml`, and `revision/phase_r6.yml`. R2 adds learned-static, parameter-matched, mean-gate, shuffled-gate, SKNet-1D, and AFNet controls. R6 repeats the five retrained models with fixed epoch 100 checkpoints and forbids validation-based early stopping or checkpoint selection. C1 and C4 remain unresolved; the configurations do not authorize claims of independent content-matching benefit or superiority over learned-static, SKNet-1D, or AFNet.
