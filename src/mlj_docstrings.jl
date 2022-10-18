
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
using MLJ
CatBoost = @load CatBoostRegressor pkg=CatBoost
model = CatBoostRegressor()

X, y = make_regression(100, 2) # synthetic data
mach = machine(model, X, y) |> fit!

Xnew, _ = make_regression(3, 2)
yhat = predict(mach, Xnew) # new predictions

fitted_params(mach).model # raw Python object for the model
```

See also
[catboost](https://github.com/catboost/catboost) and
the unwrapped model type
[`CatBoost.CatBoostRegressor`](@ref).

"""
CatBoostRegressor
