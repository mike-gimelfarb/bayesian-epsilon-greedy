# Adaptive Epsilon-Greedy Exploration Policy using Bayesian Ensembles

## Description

This small and fairly self-contained (see prerequisites below) package accompanies an article published in Uncertainty in Artificial Intelligence (UAI 2019) entitled "ε-BMC: A Bayesian Ensemble Approach to Epsilon-Greedy Exploration in
Model-Free Reinforcement Learning"

It contains an implementation of an adaptive epsilon-greedy exploration policy that adapts the exploration parameter from data in model-free reinforcement learning. 

## Prerequisites

Tested on Python 3.5 with standard packages (e.g. numpy, scipy, abc) and the following additional packages:

1. Keras with tensorflow backend
2. OpenAI Gym for the Cartpole implementation

## Citation

To cite the framework:

> @inproceedings{DBLP:conf/uai/GimelfarbSL19,
>  author    = {Michael Gimelfarb and
>               Scott Sanner and
>               Chi{-}Guhn Lee},
>  editor    = {Amir Globerson and
>               Ricardo Silva},
>  title     = {Epsilon-BMC: {A} Bayesian Ensemble Approach to Epsilon-Greedy Exploration
>               in Model-Free Reinforcement Learning},
>  booktitle = {Proceedings of the Thirty-Fifth Conference on Uncertainty in Artificial
>               Intelligence, {UAI} 2019, Tel Aviv, Israel, July 22-25, 2019},
>  pages     = {162},
>  publisher = {{AUAI} Press},
>  year      = {2019},
>  url       = {http://auai.org/uai2019/proceedings/papers/162.pdf},
>  timestamp = {Fri, 19 Jul 2019 13:05:12 +0200},
>  biburl    = {https://dblp.org/rec/conf/uai/GimelfarbSL19.bib},
>  bibsource = {dblp computer science bibliography, https://dblp.org}
> }
