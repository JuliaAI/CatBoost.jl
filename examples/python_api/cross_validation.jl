# Performs cross validation using CatBoost's built-in CV system via `cv`
module CrossValidation

using CatBoost
using DataFrames
using PythonCall

cv_data = pylist([["France", 1924, 44], ["USA", 1932, 37], ["Switzerland", 1928, 25],
                  ["Norway", 1952, 30], ["Japan", 1972, 35], ["Mexico", 1968, 112]])

labels = pylist([1, 1, 0, 0, 0, 1])

cat_features = pylist([0])

cv_dataset = Pool(; data=cv_data, label=labels, cat_features=cat_features)

params = PyDict(Dict("iterations" => 100, "depth" => 2, "loss_function" => "Logloss",
                     "verbose" => false))

scores = cv(cv_dataset; fold_count=2, params)

display(scores)

end # module
