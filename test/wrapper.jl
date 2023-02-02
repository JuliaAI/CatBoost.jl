
@testset "Python Wrapper" begin
    using CatBoost
    @testset "`to_pandas` and `pandas_to_tbl`" begin
        tbl = Tables.columntable((floats=0.5:0.5:3.0, ints=1:6))
        pd_df = CatBoost.to_pandas(tbl)
        @test pd_df isa Py
        tbl2 = pandas_to_tbl(pd_df)
        @test tbl2 == tbl
    end

    @testset "feature_importance" begin
        train_data = PyList([[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]])
        train_labels = PyList([10, 20, 30])
        model = catboost.CatBoostRegressor(; iterations=2, learning_rate=1, depth=2,
                                           verbose=false)
        CatBoost.fit!(model, train_data, train_labels)
        feat_importance = CatBoost.feature_importance(model)
        @test size(feat_importance, 1) == 4
    end

    @testset "Python Wrapper Examples" begin
        for ex in readdir(WRAPPER_EXAMPLES_DIR)
            @testset "$ex" begin
                # Just check the examples run, for now.
                include(joinpath(WRAPPER_EXAMPLES_DIR, ex))
            end
        end
    end
end
