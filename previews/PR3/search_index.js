var documenterSearchIndex = {"docs":
[{"location":"#API-Documentation-1","page":"API Documentation","title":"API Documentation","text":"","category":"section"},{"location":"#","page":"API Documentation","title":"API Documentation","text":"Below is the API documentation for CatBoost.jl.","category":"page"},{"location":"#","page":"API Documentation","title":"API Documentation","text":"For a nice introduction to the package, see the examples.","category":"page"},{"location":"#","page":"API Documentation","title":"API Documentation","text":"CurrentModule = CatBoost","category":"page"},{"location":"#","page":"API Documentation","title":"API Documentation","text":"Modules = [CatBoost]","category":"page"},{"location":"#CatBoost.Pool-Tuple{Any}","page":"API Documentation","title":"CatBoost.Pool","text":"Pool(data; label=nothing, cat_features=nothing, text_features=nothing,\n     pairs=nothing, delimiter='\t', has_header=false, weight=nothing,\n     group_id = nothing, group_weight=nothing, subgroup_id=nothing,\n     pairs_weight=nothing, baseline=nothing, features_names=nothing,\n     thread_count = -1) -> PyObject\n\nCreates a Pool object holding training data and labels. data may also be passed as a keyword argument.\n\n\n\n\n\n","category":"method"},{"location":"#CatBoost.cv","page":"API Documentation","title":"CatBoost.cv","text":"cv(pool::PyObject; kwargs...) -> DataFrame\n\nAccepts a CatBoost.Pool positional argument to specify the training data, and keyword arguments to configure the settings. See the python documentation below for what keyword arguments are accepted.\n\n\n\nPython documentation for catboost.cv\n\nCross-validate the CatBoost model.\n\nParameters\n----------\npool : catboost.Pool\n    Data to cross-validate on.\n\nparams : dict\n    Parameters for CatBoost.\n    CatBoost has many of parameters, all have default values.\n    If  None, all params still defaults.\n    If  dict, overriding some (or all) params.\n\ndtrain : catboost.Pool or tuple (X, y)\n    Synonym for pool parameter. Only one of these parameters should be set.\n\niterations : int\n    Number of boosting iterations. Can be set in params dict.\n\nnum_boost_round : int\n    Synonym for iterations. Only one of these parameters should be set.\n\nfold_count : int, optional (default=3)\n    The number of folds to split the dataset into.\n\nnfold : int\n    Synonym for fold_count.\n\ntype : string, optional (default='Classical')\n    Type of cross-validation\n    Possible values:\n        - 'Classical'\n        - 'Inverted'\n        - 'TimeSeries'\n\ninverted : bool, optional (default=False)\n    Train on the test fold and evaluate the model on the training folds.\n\npartition_random_seed : int, optional (default=0)\n    Use this as the seed value for random permutation of the data.\n    Permutation is performed before splitting the data for cross validation.\n    Each seed generates unique data splits.\n\nseed : int, optional\n    Synonym for partition_random_seed. This parameter is deprecated. Use\n    partition_random_seed instead.\n    If both parameters are initialised partition_random_seed parameter is\n    ignored.\n\nshuffle : bool, optional (default=True)\n    Shuffle the dataset objects before splitting into folds.\n\nlogging_level : string, optional (default=None)\n    Possible values:\n        - 'Silent'\n        - 'Verbose'\n        - 'Info'\n        - 'Debug'\n\nstratified : bool, optional (default=None)\n    Perform stratified sampling. True for classification and False otherwise.\n\nas_pandas : bool, optional (default=True)\n    Return pd.DataFrame when pandas is installed.\n    If False or pandas is not installed, return dict.\n\nmetric_period : int, [default=1]\n    The frequency of iterations to print the information to stdout. The value should be a positive integer.\n\nverbose : bool or int\n    If verbose is bool, then if set to True, logging_level is set to Verbose,\n    if set to False, logging_level is set to Silent.\n    If verbose is int, it determines the frequency of writing metrics to output and\n    logging_level is set to Verbose.\n\nverbose_eval : bool or int\n    Synonym for verbose. Only one of these parameters should be set.\n\nplot : bool, optional (default=False)\n    If True, draw train and eval error in Jupyter notebook\n\nearly_stopping_rounds : int\n    Activates Iter overfitting detector with od_wait set to early_stopping_rounds.\n\nsave_snapshot : bool, [default=None]\n    Enable progress snapshotting for restoring progress after crashes or interruptions\n\nsnapshot_file : string or pathlib.Path, [default=None]\n    Learn progress snapshot file path, if None will use default filename\n\nsnapshot_interval: int, [default=600]\n    Interval between saving snapshots (seconds)\n\nmetric_update_interval: float, [default=0.5]\n    Interval between updating metrics (seconds)\n\nfolds: generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)\n    If generator or iterator, it should yield the train and test indices for each fold.\n    If object, it should be one of the scikit-learn splitter classes\n    (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)\n    and have ``split`` method.\n    if folds is not None, then all of fold_count, shuffle, partition_random_seed, inverted are None\n\nReturns\n-------\ncv results : pandas.core.frame.DataFrame with cross-validation results\n    columns are: test-error-mean  test-error-std  train-error-mean  train-error-std\n\n\n\n\n\n","category":"function"},{"location":"#CatBoost.to_catboost-Tuple{Any}","page":"API Documentation","title":"CatBoost.to_catboost","text":"to_catboost(arg)\n\nto_catboost is called on each argument passed to fit, predict, predict_proba, and cv to allow customization of the conversion of Julia types to python types. If to_catboost emits a Julia type, then PyCall will try to convert it appropriately (automatically).\n\nBy default, to_catboost simply checks if the argument satisfies Tables.istable(arg), and if so, it outputs a corresponding pandas table, and otherwise passes it on.\n\nTo customize the conversion for custom types, provide a method for this function.\n\n\n\n\n\n","category":"method"}]
}
