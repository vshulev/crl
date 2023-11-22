# crl

The code in this repo is a modified version of [run_pplm.py](https://github.com/uber-research/PPLM/blob/master/run_pplm.py). Instructions to run are similar to the instructions described in the official [PPLM repository](https://github.com/uber-research/PPLM), specifically the instructions related to the BoW model (the classifier model is currently not supported with PPO).

## Modifications

The main modifications to the original PPLM code happen in the function `perturb_past`. The objective function specifically is computed after line 239 (the comment `# Compute the Objective`). To change how the reward is computed the value of the variable `rewards` can be replaced with a custom reward model. `rewards` should be a 1D tensor of shape the size of the LLM vocabulary, and should contain the rewards for each next action given the context so far.
