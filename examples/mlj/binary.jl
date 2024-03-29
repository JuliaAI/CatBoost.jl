module Binary

using CatBoost.MLJCatBoostInterface
using DataFrames
using MLJBase

# Initialize data
train_data = DataFrame([coerce(["a", "a", "c"], Multiclass),
                        coerce(["b", "b", "d"], Multiclass),
                        coerce([0, 0, 1], OrderedFactor), [4, 5, 40], [5, 6, 50],
                        [6, 7, 60]], :auto)
train_labels = coerce([1, 1, -1], OrderedFactor)
eval_data = DataFrame([coerce(["a", "a"], Multiclass), coerce(["b", "d"], Multiclass),
                       coerce([0, 0], OrderedFactor), [4, 4], [6, 50], [8, 60]], :auto)

# Initialize CatBoostClassifier
model = CatBoostClassifier(; iterations=2, learning_rate=1.0, depth=2)
mach = machine(model, train_data, train_labels)

# Fit model
MLJBase.fit!(mach)

# Get predicted classes
preds_class = MLJBase.predict_mode(mach, eval_data)

# Get predicted probabilities for each class
preds_proba = MLJBase.predict(mach, eval_data)

end # module
