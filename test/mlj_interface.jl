
@testset "CatBoostClassifier" begin
    X = DataFrame(; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
    y = [0, 0, 1, 1]

    # MLJ Interface
    model = CatBoost.CatBoostClassifier(; iterations=5)
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

function test_func(X::Union{MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor),
                            AbstractMatrix{MMI.Continuous}})
    return X
end
