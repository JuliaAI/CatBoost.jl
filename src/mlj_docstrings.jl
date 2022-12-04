
const CATBOOST_DOCS_LINK = "https://catboost.ai/en/docs/"
const CATBOOST_PARAMS_DOCS_LINK = "https://catboost.ai/en/docs/references/training-parameters/"

"""
$(MMI.doc_header(CatBoostRegressor))

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`; check column scitypes with `schema(X)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters
#TODO: Document hyper-parameters
https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier#parameters

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above.

- `predict_mean(mach, Xnew)`: return predicted probabilities of the 
  target given new features `Xnew` having the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: The Python CatBoostClassifier model


# Report

- `feature_importances`: DataFrame of feature importances


# Examples

```
using CatBoost
using DataFrames
using MLJBase

X = DataFrame(; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
y = [0, 0, 1, 1]

model = CatBoost.CatBoostClassifier(iterations=5)
mach = machine(model, X, y)
MLJBase.fit!(mach)
probs = MLJBase.predict(mach, X)
preds = MLJBase.predict_mean(mach, X)
```

See also
[catboost](https://github.com/catboost/catboost) and
the unwrapped model type
[`CatBoost.CatBoostClassifier`](@ref).

"""
CatBoostClassifier

"""
$(MMI.doc_header(CatBoostRegressor))

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`; check column scitypes with `schema(X)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters
#TODO: Document hyper-parameters
https://catboost.ai/en/docs/concepts/python-reference_catboostregressor#parameters

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: The Python CatBoostRegressor model


# Report

- `feature_importances`: DataFrame of feature importances


# Examples

```
using CatBoost
using DataFrames
using MLJBase

X = DataFrame(; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
y = [2.0, 4.0, 6.0, 7.0]

model = CatBoostRegressor(iterations=5)
mach = machine(model, X, y)
MLJBase.fit!(mach)
preds = MLJBase.predict(mach, X)
```

See also
[catboost](https://github.com/catboost/catboost) and
the unwrapped model type
[`CatBoost.CatBoostRegressor`](@ref).

"""
CatBoostRegressor
