module Binary

using CatBoost.MLJCatBoostInterface
using DataFrames
using MLJBase
using PythonCall

# Initialize data
cat_features = [0, 1]
train_data = DataFrame([["a", "a", "c"], ["b", "b", "d"], [1, 4, 30], [4, 5, 40],
                        [5, 6, 50], [6, 7, 60]], :auto)
train_labels = [1, 1, -1]
eval_data = DataFrame([["a", "a"], ["b", "d"], [2, 1], [4, 4], [6, 50], [8, 60]], :auto)

# Initialize CatBoostClassifier
model = CatBoostClassifier(; iterations=2, learning_rate=1, depth=2,
                           cat_features=cat_features)
mach = machine(model, train_data, train_labels)

# Fit model
MLJBase.fit!(mach)

# Get predicted classes
preds_class = MLJBase.predict(mach, eval_data)

# Get predicted probabilities for each class
preds_proba = MLJBase.predict_mean(mach, eval_data)

end # module
