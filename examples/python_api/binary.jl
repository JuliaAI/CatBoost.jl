module Binary

using CatBoost
using DataFrames
using PythonCall

# Initialize data
cat_features = pylist([0, 1])
train_data = DataFrame([["a", "a", "c"], ["b", "b", "d"], [1, 4, 30], [4, 5, 40],
                        [5, 6, 50], [6, 7, 60]], :auto)
train_labels = pylist([1, 1, -1])
eval_data = DataFrame([["a", "a"], ["b", "d"], [2, 1], [4, 4], [6, 50], [8, 60]], :auto)

# Initialize CatBoostClassifier
model = PyCatBoostClassifier(; iterations=2, learning_rate=1, depth=2)
# Fit model
fit!(model, train_data, train_labels, cat_features)

# Get predicted classes
preds_class = predict(model, eval_data)

# Get predicted probabilities for each class
preds_proba = predict_proba(model, eval_data)

# Get predicted RawFormulaVal
preds_raw = predict(model, eval_data; prediction_type="RawFormulaVal")

end # module
