module MultiClass

using CatBoost
using PythonCall

train_data = PyList([["summer", 1924, 44], ["summer", 1932, 37], ["winter", 1980, 37],
                     ["summer", 2012, 204]])

eval_data = PyList([["winter", 1996, 197], ["winter", 1968, 37], ["summer", 2002, 77],
                    ["summer", 1948, 59]])

cat_features = PyList([0])

train_label = PyList(["France", "USA", "USA", "UK"])
eval_label = PyList(["USA", "France", "USA", "UK"])

train_dataset = Pool(; data=train_data, label=train_label, cat_features=cat_features)

eval_dataset = Pool(; data=eval_data, label=eval_label, cat_features=cat_features)

# Initialize CatBoostClassifier
model = CatBoostClassifier(; iterations=10, learning_rate=1, depth=2,
                           loss_function="MultiClass")

# Fit model
fit!(model, train_dataset)

# Get predicted classes
preds_class = predict(model, eval_dataset)

# Get predicted probabilities for each class
preds_proba = predict_proba(model, eval_dataset)

# Get predicted RawFormulaVal
preds_raw = predict(model, eval_dataset; prediction_type="RawFormulaVal")

end # module
