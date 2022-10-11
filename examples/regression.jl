module Regression

using CatBoost
using PythonCall

train_data = [[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]] |> PyList
eval_data = [[2, 4, 6, 8], [1, 4, 50, 60]] |> PyList
train_labels = [10, 20, 30] |> PyList

# Initialize CatBoostRegressor
model = CatBoostRegressor(; iterations=2, learning_rate=1, depth=2)

# Fit model
fit!(model, train_data, train_labels)

# Get predictions
preds = predict(model, eval_data)

end # module
