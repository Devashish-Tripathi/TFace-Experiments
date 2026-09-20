<!-- <img src="recognition/doc/logo.png" title="Logo" width="600" /> -->

## Rough Overview

DCTDP modification from Tencent TFace Library.

Baseline:

**`2022.9`**: `Privacy-Preserving Face Recognition with Learnable Privacy Budgets in Frequency Domain` accepted by **ECCV2022**. 
[[paper](https://arxiv.org/abs/2207.07316)]

## License and Attribution
This repository is a modified fork of [Tencent/TFace](https://github.com/Tencent/TFace), originally released under the Apache 2.0 License. 

Modifications made for this project:
- Pruned unused tasks, backbones, and detection modules.
- Isolated and adapted the DCTDP pipeline for standalone execution.
- Updated data-loading integrations (dareblopy-dx).

See `LICENSE.txt` for the original license text and third-party notices.

Currently WIP. 