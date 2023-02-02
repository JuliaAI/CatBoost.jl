
const CATBOOST_DOCS_LINK = "https://catboost.ai/en/docs/"
const CATBOOST_PARAMS_DOCS_LINK = "https://catboost.ai/en/docs/references/training-parameters/"

"""
$(MMI.doc_header(CatBoostClassifier))

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, `Finite`, `Textual`; check column scitypes with `schema(X)`.
  `Textual` columns will be passed to catboost as `text_features`,
  `Multiclass` columns will be passed to catboost as `cat_features`, and
  `OrderedFactor` columns will be converted to integers.

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Finite`; check the scitype with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.

# Hyper-parameters

More details on the catboost hyperparameters, here are the Python docs: 
https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier#parameters

# Operations

- `predict(mach, Xnew)`: probabilistic predictions of the target given new
  features `Xnew` having the same scitype as `X` above.

- `predict_mode(mach, Xnew)`: returns the mode of each of the prediction above.

# Accessor functions

- `feature_importances(mach)`: return vector of feature importances, in the form of  
  `feature::Symbol => importance::Real` pairs

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: The Python CatBoostClassifier model

# Report

The fields of `report(mach)` are:
- `feature_importances`: Vector{Pair{Symbol, Float64}} of feature importances

# Examples

```
using CatBoost.MLJCatBoostInterface
using MLJ

X = (
    duration = [1.5, 4.1, 5.0, 6.7], 
    n_phone_calls = [4, 5, 6, 7], 
    department = coerce(["acc", "ops", "acc", "ops"], Multiclass), 
)
y = coerce([0, 0, 1, 1], Multiclass)

model = CatBoostClassifier(iterations=5)
mach = machine(model, X, y)
fit!(mach)
probs = predict(mach, X)
preds = predict_mode(mach, X)
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
  `Count`, `Finite`, `Textual`; check column scitypes with `schema(X)`.
  `Textual` columns will be passed to catboost as `text_features`,
  `Multiclass` columns will be passed to catboost as `cat_features`, and
  `OrderedFactor` columns will be converted to integers.

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.

# Hyper-parameters

More details on the catboost hyperparameters, here are the Python docs: 
https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier#parameters

# Operations

- `predict(mach, Xnew)`: probabilistic predictions of the target given new
  features `Xnew` having the same scitype as `X` above.

# Accessor functions

- `feature_importances(mach)`: return vector of feature importances, in the form of  
  `feature::Symbol => importance::Real` pairs

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: The Python CatBoostRegressor model

# Report

The fields of `report(mach)` are:
- `feature_importances`: Vector{Pair{Symbol, Float64}} of feature importances

# Examples

```
using CatBoost.MLJCatBoostInterface
using MLJ

X = (
    duration = [1.5, 4.1, 5.0, 6.7], 
    n_phone_calls = [4, 5, 6, 7], 
    department = coerce(["acc", "ops", "acc", "ops"], Multiclass), 
)
y = [2.0, 4.0, 6.0, 7.0]

model = CatBoostRegressor(iterations=5)
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```

See also
[catboost](https://github.com/catboost/catboost) and
the unwrapped model type
[`CatBoost.CatBoostRegressor`](@ref).

"""
CatBoostRegressor
