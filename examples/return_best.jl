module ReturnBestOnMetric

using CatBoost
using PythonCall

train_data = [[0, 3], [4, 1], [8, 1], [9, 1]] |> PyList

train_labels = [0, 0, 1, 1] |> PyList

eval_data = [[2, 1], [3, 1], [9, 0], [5, 3]] |> PyList

eval_labels = [0, 1, 1, 0] |> PyList

eval_dataset = Pool(; data=eval_data, label=eval_labels)

model = CatBoostClassifier(; learning_rate=0.03,
                           custom_metric=["Logloss", "AUC:hints=skip_train~false"])
fit!(model, train_data, train_labels; eval_set=eval_dataset, verbose=false)

for (k, v) in model.get_best_score()
    println("$k : $v")
end

end # module
