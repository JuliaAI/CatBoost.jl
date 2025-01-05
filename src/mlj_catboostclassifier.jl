
MMI.@mlj_model mutable struct CatBoostClassifier <: MMI.Probabilistic
    iterations::Int = 1000::(_ > 0)
    learning_rate::Float64 = 0.03::(_ >= 0)
    depth::Int = 6::(_ > 0)
    l2_leaf_reg::Float64 = 3.0::(_ >= 0)
    model_size_reg::Union{Float64} = 0.5
    rsm::Float64 = 1.0::((_ > 0) & (_ <= 1.0))
    loss_function::Union{String,Nothing} = nothing
    border_count::Union{Int,Nothing} = nothing
    feature_border_type::Union{String,Nothing} = nothing
    per_float_feature_quantization::Union{Py,Nothing} = nothing
    input_borders::Union{String,Nothing} = nothing
    output_borders::Union{String,Nothing} = nothing
    fold_permutation_block::Int = 1::(_ > 0)
    nan_mode::String = "Min"::(_ in ("Forbidden", "Min", "Max"))
    counter_calc_method::String = "SkipTest"::(_ in ("Full", "SkipTest"))
    leaf_estimation_iterations::Union{Int,Nothing} = nothing
    leaf_estimation_method::Union{String,Nothing} = nothing
    thread_count::Int = -1
    random_seed::Union{Int,Nothing} = nothing
    metric_period::Int = 1
    ctr_leaf_count_limit::Union{Int,Nothing} = nothing
    store_all_simple_ctr::Bool = false
    max_ctr_complexity::Union{Bool,Nothing} = nothing
    has_time::Bool = false
    allow_const_label::Bool = false
    target_border::Union{Float64,Nothing} = nothing
    class_weights::Union{Py,Nothing} = nothing
    auto_class_weights::Union{Bool,Nothing} = nothing
    one_hot_max_size::Union{Int,Nothing} = nothing
    random_strength::Float64 = 1.0
    bagging_temperature::Float64 = 1.0
    fold_len_multiplier::Float64 = 2.0
    used_ram_limit::Union{Int,Nothing} = nothing
    gpu_ram_part::Float64 = 0.95::((_ > 0) & (_ <= 1.0))
    pinned_memory_size::Int = 1073741824
    allow_writing_files::Union{Bool,Nothing} = nothing
    approx_on_full_history::Bool = false
    boosting_type::Union{String,Nothing} = nothing
    simple_ctr::Union{Py,Nothing} = nothing
    combinations_ctr::Union{Py,Nothing} = nothing
    per_feature_ctr::Union{Py,Nothing} = nothing
    task_type::Union{String,Nothing} = nothing
    devices::Union{String,Nothing} = nothing
    bootstrap_type::Union{String,Nothing} = nothing
    subsample::Union{Int,Nothing} = nothing
    sampling_frequency::String = "PerTreeLevel"::(_ in ("PerTree", "PerTreeLevel"))
    sampling_unit::String = "Object"::(_ in ("Group", "Object"))
    gpu_cat_features_storage::String = "GpuRam"::(_ in ("CpuPinnedMemory", "GpuRam"))
    data_partition::Union{String,Nothing} = nothing
    early_stopping_rounds::Union{Int,Nothing} = nothing
    grow_policy::String = "SymmetricTree"::(_ in
                                            ("Depthwise ", "Lossguide", "SymmetricTree"))
    min_data_in_leaf::Int = 1::(_ > 0)
    max_leaves::Int = 31::(_ > 0)
    leaf_estimation_backtracking::String = "AnyImprovement"::(_ in
                                                              ("AnyImprovement", "Armijo",
                                                               "No"))
    feature_weights::Union{Nothing,Py} = nothing
    penalties_coefficient::Float64 = 1.0
    model_shrink_rate::Union{Float64,Nothing} = nothing
    model_shrink_mode::String = "Constant"::(_ in ("Constant", "Decreasing"))
    langevin::Bool = false
    diffusion_temperature::Float64 = 10_000.0::(_ >= 0)
    posterior_sampling::Bool = false
    boost_from_average::Union{Bool,Nothing} = nothing
    text_processing::Union{Py,Nothing} = nothing # https://catboost.ai/en/docs/references/text-processing__test-processing__default-value
end

function model_init(mlj_model::CatBoostClassifier; kw...)
    return catboost.CatBoostClassifier(; mlj_to_kwargs(mlj_model)..., kw...)
end

prepare_single_class(y_first) = CategoricalArray([y_first])[1]
prepare_single_class(y_first::CategoricalValue) = y_first

function MMI.fit(mlj_model::CatBoostClassifier, verbosity::Int, data_pool, y_first)
    # Check if the data pool has only one unique value
    y_first = prepare_single_class(y_first)
    unique_classes = pyconvert(Vector, numpy.unique(data_pool.get_label()))
    if length(unique_classes) == 1
        # Skip training and store the single class
        fitresult = (model=nothing, single_class=unique_classes[1], y_first=y_first)
        cache = (; mlj_model=deepcopy(mlj_model))
        report = (feature_importances=[],) # No feature importances in this case
        return (fitresult, cache, report)
    end

    verbose = verbosity <= 1 ? false : true
    model = model_init(mlj_model; verbose)
    model.fit(data_pool)

    cache = (; mlj_model=deepcopy(mlj_model))
    report = (feature_importances=feature_importance(model),)
    fitresult = (model=model, y_first=y_first)

    return (fitresult, cache, report)
end

MMI.fitted_params(::CatBoostClassifier, model) = (model=model,)
MMI.reports_feature_importances(::Type{<:CatBoostClassifier}) = true

function MMI.predict(mlj_model::CatBoostClassifier, fitresult, X_pool)
    if fitresult.model === nothing
        # Always predict the single class
        n = pyconvert(Int, X_pool.shape[0])
        classes = [fitresult.single_class]
        probs = ones(n, 1)
        return MMI.UnivariateFinite(classes, probs; pool=fitresult.y_first.pool)
    end

    classes = pyconvert(Array, fitresult.model.classes_.tolist())
    py_preds = predict_proba(fitresult.model, X_pool)
    preds = MMI.UnivariateFinite(classes, pyconvert(Array, py_preds);
                                 pool=fitresult.y_first.pool)
    return preds
end

function MMI.predict_mode(mlj_model::CatBoostClassifier, fitresult, X_pool)
    if fitresult[1] === nothing
        # Return probability 1 for the single class
        n = pyconvert(Int, X_pool.shape[0])
        return fill(fitresult.y_first, n)
    end

    model, y_first = fitresult
    py_preds = predict(model, X_pool)
    preds = pyconvert(Array, py_preds)
    return preds
end
