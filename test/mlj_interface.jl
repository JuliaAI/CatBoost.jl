
@testset "MLJ Examples" begin
    for ex in readdir(MLJ_EXAMPLES_DIR)
        @testset "$ex" begin
            # Just check the examples run, for now.
            include(joinpath(MLJ_EXAMPLES_DIR, ex))
        end
    end
end

@testset "CatBoostClassifier" begin
    X = DataFrame(; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
    y = [0, 0, 1, 1]

    # MLJ Interface
    model = CatBoostClassifier(; iterations=5)
    mach = machine(model, X, y)
    MLJBase.fit!(mach)
    preds = MLJBase.predict(mach, X)
    probs = MLJBase.predict_mean(mach, X)
end

@testset "CatBoostRegressor" begin
    X = DataFrame(; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
    y = [2.0, 4.0, 6.0, 7.0]

    # MLJ Interface
    model = CatBoostRegressor(; iterations=5)
    mach = machine(model, X, y)
    MLJBase.fit!(mach)
    preds = MLJBase.predict(mach, X)
end

@testset "generic interface tests" begin
    @testset "CatBoostRegressor" begin
        failures, summary = MLJTestInterface.test(
            [CatBoostRegressor,],
            MLJTestInterface.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "CatBoostClassifier" begin
        for data in [
            MLJTestInterface.make_binary(),
            MLJTestInterface.make_multiclass(),
        ]
            failures, summary = MLJTestInterface.test(
                [CatBoostClassifier],
                data...;
                mod=@__MODULE__,
                verbosity=0, # bump to debug
                throw=false, # set to true to debug
            )
            @test isempty(failures)
        end
    end
end
