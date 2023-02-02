module Regression

using CatBoost.MLJCatBoostInterface
using DataFrames
using MLJBase
using PythonCall

# Initialize data
train_data = DataFrame([[1, 4, 30], [4, 5, 40], [5, 6, 50], [6, 7, 60]], :auto)
train_labels = [10.0, 20.0, 30.0]
eval_data = DataFrame([[2, 1], [4, 4], [6, 50], [8, 60]], :auto)

# Initialize CatBoostClassifier
model = CatBoostRegressor(; iterations=2, learning_rate=1.0, depth=2)
mach = machine(model, train_data, train_labels)

# Fit model
MLJBase.fit!(mach)

# Get predictions
preds_class = MLJBase.predict(mach, eval_data)

end # module
