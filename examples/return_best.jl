module ReturnBestOnMetric

using CatBoost

train_data = [[0, 3], [4, 1], [8, 1], [9, 1]]

train_labels = [0, 0, 1, 1]

eval_data = [[2, 1], [3, 1], [9, 0], [5, 3]]

eval_labels = [0, 1, 1, 0]

eval_dataset = Pool(; data=eval_data, label=eval_labels)

model = CatBoostClassifier(; learning_rate=0.03,
                           custom_metric=["Logloss", "AUC:hints=skip_train~false"])
fit!(model, train_data, train_labels; eval_set=eval_dataset, verbose=false)

for (k, v) in model.get_best_score()
    println("$k : $v")
end

end # module
