# Invertible Zero-Shot Recognition Flows

This repo contains an implementation of https://arxiv.org/abs/2007.04873

Changes have been made to obtain the results of the toy_data experiments present in the article:
  - Added LinearLU
  - Added ActNorm

Mutiple runs of this implementation with different values of _centralizing_loss_ and _learning_rate_ can be found here:
https://wandb.ai/mvalente/toy_data_zf/sweeps/v4eewmtw?workspace=user-mvalente

Contains code from https://github.com/SamGalanakis/FlowCompare
