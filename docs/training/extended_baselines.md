# Extended Validation Baselines

The final comparison adds four train/validation-only baselines under the same RadioML 2016.10A split, `per_sample_max_abs` preprocessing, train-only phase/shift augmentation, AdamW budget, checkpoint rule, and five seeds as S0/S1/S2.

| Model | Role | Parameters | MACs |
| --- | --- | ---: | ---: |
| `resnet1d_macs` | Narrow ResNet1D MAC control | 794,495 | 4,495,928 |
| `mobilenetv2_1d` | MobileNetV2-style parameter control | 130,907 | 3,446,568 |
| `mcldnn` | Strong recurrent/temporal baseline | 406,199 | 48,589,568 |
| `se_msfn_1d` | Nearest multi-scale SE/fusion baseline | 91,947 | 5,401,280 |

S2 has 124,861 parameters and 4,654,792 MACs under the same counter. The MobileNetV2-style parameter gap is 4.84%; the narrow ResNet1D MAC gap is 3.41%. These widths were frozen before formal validation and must not be changed after seeing results.

MCLDNN is adapted from `HantongXING/SigDA@b68c8563687f8ffa45dc69a8238bf0703aac101f` under MIT, retaining its full-I/Q branch, separate I/Q branches, joint convolution, two-layer LSTM, and classifier topology. External data loading, random splitting, training schedule, checkpoints, and reported results are not reused.

`mobilenetv2_1d` is a one-dimensional implementation of inverted residuals, depthwise convolutions, and linear bottlenecks. `se_msfn_1d` is a source-informed adaptation of arXiv `2209.03764`, retaining kernel-9 convolutions, SE bottlenecks, multi-resolution downsampling fusion, and global pooling. The original SE-MSFN uses 1024-sample RadioML 2018.01A inputs and does not publish directly executable source for this exact adapter, so this is not called a source-aligned reproduction.

Every extended baseline first runs the frozen seed-13 grid `learning_rate={1e-3,3e-4}` and `dropout={0,0.2}`, then the selected configuration runs seeds `13/37/73/101/137`. `audit_extended_baselines.py` records the structure, configuration hashes, complexity controls, and source boundaries before formal training. Test access remains forbidden.
