
@testset "CatBoostRegressor" begin
    X = DataFrame(; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
    y = [2, 4, 6, 7]

    # MLJ Interface
    model = CatBoostRegressor()
    mach = machine(model, X, y)
    MLJBase.fit!(mach)
    preds = MLJBase.predict(mach, X)
end
