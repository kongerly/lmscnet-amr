# Final S0/S1/S2 Validation Family

The final family is separate from the historical noise-aware `NALMSCNet`. Every final model accepts only `iq` with shape `[batch, 2, 128]` and returns `{logits, scale_weights}`. No final parameter or module path contains an SNR head, SNR loss, SNR conditioning, noise embedding, or constant embedding.

All variants share the same stem, `32/64/96` stage channels, six residual blocks, classifier, max-abs preprocessing, train-only phase/shift augmentation, AdamW configuration, cosine schedule, 100-epoch budget, Macro-F1 checkpoint rule, and patience `12`. S0 trains kernels `3`, `7`, and `15` independently. S1 keeps `[3,7,15]` with fixed equal weights. S2 adds only a per-block gate over pooled branch content. The pre-registered widened S0 uses kernel `7` and expansion `1.42`.

`audit_final_experiment_family.py` verifies the real train/validation input binding, configuration hashes, output interface, forbidden-path absence, S1/S2 non-gate state-schema equality, parameters, and MACs. The verified counts are:

| Model | Parameters | MACs |
| --- | ---: | ---: |
| S0-k3 | 78,283 | 4,001,312 |
| S0-k7 | 80,203 | 4,113,952 |
| S0-k15 | 84,043 | 4,339,232 |
| S0-wide-k7 | 90,031 | 4,647,008 |
| S1 | 90,763 | 4,620,832 |
| S2 | 124,861 | 4,654,792 |

The detached validation snapshot is commit `f5760d85ff0bbcf28b1f6005f3ef5dad1e615de6`. Its queue protocol and all generated checkpoints/metrics are stored outside the repository under `<data-dir>\final-validation-f5760d8`. The queue runs the six final configurations for seeds `13/37/73/101/137`, reruns max-abs seed-13 sweeps for CNN2/CLDNN/ResNet1D, then runs the selected baseline configurations for the same five seeds. All stages keep `test_accessed=false`.
