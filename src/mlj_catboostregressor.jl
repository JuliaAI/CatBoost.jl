
MMI.@mlj_model mutable struct CatBoostRegressor <: MMI.Deterministic
    iterations = nothing
    learning_rate = nothing
    depth = nothing
    l2_leaf_reg = nothing
    model_size_reg = nothing
    rsm = nothing
    loss_function::String = "RMSE"
    border_count = nothing
    feature_border_type = nothing
    per_float_feature_quantization = nothing
    input_borders = nothing
    output_borders = nothing
    fold_permutation_block = nothing
    od_pval = nothing
    od_wait = nothing
    od_type = nothing
    nan_mode = nothing
    counter_calc_method = nothing
    leaf_estimation_iterations = nothing
    leaf_estimation_method = nothing
    thread_count = nothing
    random_seed = nothing
    use_best_model = nothing
    best_model_min_trees = nothing
    logging_level = nothing
    metric_period = nothing
    ctr_leaf_count_limit = nothing
    store_all_simple_ctr = nothing
    max_ctr_complexity = nothing
    has_time = nothing
    allow_const_label = nothing
    target_border = nothing
    one_hot_max_size = nothing
    random_strength = nothing
    name = nothing
    ignored_features = nothing
    train_dir = nothing
    custom_metric = nothing
    eval_metric = nothing
    bagging_temperature = nothing
    save_snapshot = nothing
    snapshot_file = nothing
    snapshot_interval = nothing
    fold_len_multiplier = nothing
    used_ram_limit = nothing
    gpu_ram_part = nothing
    pinned_memory_size = nothing
    allow_writing_files = nothing
    final_ctr_computation_mode = nothing
    approx_on_full_history = nothing
    boosting_type = nothing
    simple_ctr = nothing
    combinations_ctr = nothing
    per_feature_ctr = nothing
    ctr_description = nothing
    ctr_target_border_count = nothing
    task_type = nothing
    device_config = nothing
    devices = nothing
    bootstrap_type = nothing
    subsample = nothing
    mvs_reg = nothing
    sampling_frequency = nothing
    sampling_unit = nothing
    dev_score_calc_obj_block_size = nothing
    dev_efb_max_buckets = nothing
    sparse_features_conflict_fraction = nothing
    max_depth = nothing
    n_estimators = nothing
    num_boost_round = nothing
    num_trees = nothing
    colsample_bylevel = nothing
    random_state = nothing
    reg_lambda = nothing
    objective = nothing
    eta = nothing
    max_bin = nothing
    gpu_cat_features_storage = nothing
    data_partition = nothing
    metadata = nothing
    early_stopping_rounds = nothing
    grow_policy = nothing
    min_data_in_leaf = nothing
    min_child_samples = nothing
    max_leaves = nothing
    num_leaves = nothing
    score_function = nothing
    leaf_estimation_backtracking = nothing
    ctr_history_unit = nothing
    monotone_constraints = nothing
    feature_weights = nothing
    penalties_coefficient = nothing
    first_feature_use_penalties = nothing
    per_object_feature_penalties = nothing
    model_shrink_rate = nothing
    model_shrink_mode = nothing
    langevin = nothing
    diffusion_temperature = nothing
    posterior_sampling = nothing
    boost_from_average = nothing
    tokenizers = nothing
    dictionaries = nothing
    feature_calcers = nothing
    text_processing = nothing
    embedding_features = nothing
    eval_fraction = nothing
end

function model_init(mlj_model::CatBoostRegressor; kw...)
    return catboost.CatBoostRegressor(; mlj_to_kwargs(mlj_model)..., kw...)
end

function MMI.fit(mlj_model::CatBoostRegressor, verbosity::Int, X, y)
    verbose = verbosity > 0 ? false : true

    X_preprocessed, cat_features, text_features = prepare_input(X)
    py_X = to_pandas(X_preprocessed)
    py_y = numpy.array(y)

    model = model_init(mlj_model; cat_features, text_features, verbose)
    model.fit(py_X, py_y)

    cache = nothing
    report = (feature_importances=feature_importances(model),)

    return (model, cache, report)
end

MMI.fitted_params(::CatBoostRegressor, model) = (model=model,)
MMI.reports_feature_importances(::Type{<:CatBoostRegressor}) = true

function MMI.predict(mlj_model::CatBoostRegressor, model, Xnew)
    X_preprocessed, _, _ = prepare_input(Xnew)
    py_preds = predict(model, to_pandas(X_preprocessed))
    preds = pyconvert(Array, py_preds)
    return preds
end
