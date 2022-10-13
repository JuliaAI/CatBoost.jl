# CatBoost.jl

[![Build Status][build-img]][build-url] [![CodeCov][codecov-img]][codecov-url] [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://beacon-biosignals.github.io/CatBoost.jl/dev)

[build-img]: https://github.com/beacon-biosignals/CatBoost.jl/workflows/CI/badge.svg
[build-url]: https://github.com/beacon-biosignals/CatBoost.jl/actions
[codecov-img]: https://codecov.io/gh/beacon-biosignals/CatBoost.jl/branch/main/graph/badge.svg?token=e4RFBNkB9a
[codecov-url]: https://codecov.io/github/beacon-biosignals/CatBoost.jl


Julia interface to [CatBoost](https://catboost.ai/).

## Example

```julia
module Regression

using CatBoost

train_data = PyList([[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]])
eval_data = PyList([[2, 4, 6, 8], [1, 4, 50, 60]])
train_labels = PyList([10, 20, 30])

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations = 2, learning_rate = 1, depth = 2)

# Fit model
fit!(model, train_data, train_labels)

# Get predictions
preds = predict(model, eval_data)

end # module
```
