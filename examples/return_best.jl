module ReturnBestOnMetric

using CatBoost
using PythonCall

train_data = PyList([[0, 3], [4, 1], [8, 1], [9, 1]])

train_labels = PyList([0, 0, 1, 1])

eval_data = PyList([[2, 1], [3, 1], [9, 0], [5, 3]])

eval_labels = PyList([0, 1, 1, 0])

eval_dataset = Pool(; data=eval_data, label=eval_labels)

model = PyCatBoostClassifier(; learning_rate=0.03,
                             custom_metric=["Logloss", "AUC:hints=skip_train~false"])
fit!(model, train_data, train_labels; eval_set=eval_dataset, verbose=false)

for (k, v) in model.get_best_score()
    println("$k : $v")
end

end # module
