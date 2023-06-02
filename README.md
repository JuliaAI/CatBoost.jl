# CatBoost.jl

[![Build Status][build-img]][build-url] [![CodeCov][codecov-img]][codecov-url] [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaAI.github.io/CatBoost.jl/dev)

[build-img]: https://github.com/JuliaAI/CatBoost.jl/workflows/CI/badge.svg
[build-url]: https://github.com/JuliaAI/CatBoost.jl/actions
[codecov-img]: https://codecov.io/gh/JuliaAI/CatBoost.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/github/JuliaAI/CatBoost.jl


Julia interface to [CatBoost](https://catboost.ai/).

## Example

```julia
module Regression

using CatBoost
using PythonCall

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

## MLJ Example
```julia
module Regression

using CatBoost
using DataFrames
using MLJBase

train_data = DataFrame([[1,4,30], [4,5,40], [5,6,50], [6,7,60]], :auto)
eval_data = DataFrame([[2,1], [4,4], [6,50], [8,60]], :auto)
train_labels = [10.0, 20.0, 30.0] 

# Initialize MLJ Machine
model = CatBoostRegressor(iterations = 2, learning_rate = 1, depth = 2)
mach = machine(model, train_data, train_labels)

# Fit model
MLJBase.fit!(mach)

# Get predictions
preds = predict(model, eval_data)

end # module
```

# Restricting Python catboost version

By default, `CatBoost.jl` installs the latest compatible version of `catboost` (version `>=1.1`) in your current `CondaPkg.jl` environment. To install a specific version, create a `CondaPkg.toml` file using `CondaPkg.jl`. Below is an example for specifying `catboost` version `v1.1`:

```julia
using CondaPkg
CondaPkg.add("catboost"; version="=1.1")
```

This will create a `CondaPkg.toml` file in your current envrionment with the restricted `catboost` version. For more information on managing Conda environments with `CondaPkg.jl`, refer to the [official documentation](https://github.com/cjdoris/CondaPkg.jl).
