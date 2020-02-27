# causal-forest

[![Contributors][contributors-badge]][contributors-url]
[![MIT License][license-badge]][license-url]
[![Coverage][coverage-badge]][coverage-url]
[![Build Status][build-badge]][build-url]
[![Documentation Status][documentation-badge]][documentation-url]
[![Black Code Style][black-badge]][black-url]

## Word of warning

The package is still in development stage and should not be used other than for
experimental reasons.
With version 0.1.0 we will benchmark our code against existing code.

## Introduction

The ``cforest`` package can be used to estimate heterogeneous treatment effects
in a [Neyman-Rubin potential outcome framework](https://en.wikipedia.org/wiki/Rubin_causal_model).
It implements the Causal Forest algorithm first formulated in Athey and Wager (2018).


## Install

The package can be installed via conda. To do so, type the following commands in a terminal:

```console
conda install -c timmens cforest
```


## Documentation

The documentation is hosted at https://causal-forest.readthedocs.io/en/latest/.


## Example

### Complete example:

For a complete working example going through all main features please view our example notebook.
<a href="https://nbviewer.jupyter.org/github/timmens/causal-forest/blob/master/docs/source/getting_started/example.ipynb"
   target="_parent">
   <img align="center" 
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
      width="109" height="20">
</a>

### Minimal example:

```python
from cforest.forest import CausalForest

X, t, y = simulate_data()

cf = CausalForest()

cf = cf.fit(X, t, y)

XX = simulate_new_features()
predictions = cf.predict(XX)
```

## References

- Athey and Imbens, 2016, [Estimation and Inference of Heterogeneous Treatment Effects using Random Forests](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839)

- Athey and Wager, 2019, [Recursive partitioning for heterogeneous causal effects](https://www.pnas.org/content/113/27/7353)

- Athey, Tibshirani and Wager, 2019, [Generalized random forests](https://projecteuclid.org/euclid.aos/1547197251)



[contributors-badge]: https://img.shields.io/github/contributors/timmens/causal-forest
[contributors-url]: https://github.com/timmens/causal-forest/graphs/contributors
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/timmens/causal-forest/blob/master/LICENSE
[build-badge]: https://travis-ci.org/timmens/causal-forest.svg?branch=master
[build-url]: https://travis-ci.org/timmens/causal-forest
[coverage-badge]:https://codecov.io/gh/timmens/causal-forest/branch/master/graph/badge.svg
[coverage-url]:https://codecov.io/gh/timmens/causal-forest
[documentation-badge]:https://readthedocs.org/projects/causal-forest/badge/?version=latest
[documentation-url]:https://causal-forest.readthedocs.io/en/latest/?badge=latest
[black-badge]:https://img.shields.io/badge/code%20style-black-000000.svg
[black-url]:https://github.com/psf/black
