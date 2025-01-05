
@testset "MLJ Interface" begin
    using CatBoost.MLJCatBoostInterface: CatBoostRegressor, CatBoostClassifier

    @testset "MLJ Examples" begin
        for ex in readdir(MLJ_EXAMPLES_DIR)
            @testset "$ex" begin
                # Just check the examples run, for now.
                include(joinpath(MLJ_EXAMPLES_DIR, ex))
            end
        end
    end

    @testset "CatBoostClassifier" begin
        X = (; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
        y = coerce([0, 0, 1, 1], OrderedFactor)

        # MLJ Interface
        model = CatBoostClassifier(; iterations=5)
        mach = machine(model, X, y)
        MLJBase.fit!(mach)
        preds = MLJBase.predict(mach, X)
        probs = MLJBase.predict_mode(mach, X)

        serializable_fitresult = MLJBase.save(mach, mach)
        restored_fitresult = MLJBase.restore(mach, serializable_fitresult)
    end

    @testset "CatBoostClassifier - single class" begin
        X = (; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
        y = [0, 0, 0, 0]

        # MLJ Interface
        model = CatBoostClassifier(; iterations=5)
        mach = machine(model, X, y)
        MLJBase.fit!(mach)
        preds = MLJBase.predict(mach, X)
        println(preds)
        probs = MLJBase.predict_mode(mach, X)
        println(probs)

        serializable_fitresult = MLJBase.save(mach, mach)
        restored_fitresult = MLJBase.restore(mach, serializable_fitresult)
    end

    @testset "CatBoostRegressor" begin
        X = (; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
        y = [2.0, 4.0, 6.0, 7.0]

        # MLJ Interface
        model = CatBoostRegressor(; iterations=5)
        mach = machine(model, X, y)
        MLJBase.fit!(mach)
        preds = MLJBase.predict(mach, X)

        serializable_fitresult = MLJBase.save(mach, mach)
        restored_fitresult = MLJBase.restore(mach, serializable_fitresult)
    end

    @testset "evaluate" begin
        using Statistics

        X = (a=[1, 2, 3, 4], b=[4, 5, 6, 7])
        y = [1.0, 2.0, 3.0, 4.0]
        model = CatBoostRegressor(; iterations=5)
        mach = machine(model, X, y)
        e = evaluate!(mach; resampling=Holdout(; fraction_train=0.7),
                      measure=l1, verbosity=0)
        @test e.measurement[1] >= 0.0 # l1 should be non-negative

        y_class = categorical(["cat", "dog", "cat", "dog"])
        model_class = CatBoostClassifier(; iterations=5)
        mach_class = machine(model_class, X, y_class)
        e_class = evaluate!(mach_class; resampling=CV(; nfolds=2),
                            measure=accuracy, verbosity=0)
        @test 0.0 <= e_class.measurement[1] <= 1.0 # accuracy between 0 and 1
    end

    @testset "MLJ GridSearch" begin
        X = (a=[1, 2, 3, 4, 5, 6], b=[4, 5, 6, 7, 8, 9])
        y = [1.0, 2.0, 3.0, 4.0, 5, 6]
        model = CatBoostRegressor()
        r = range(model, :iterations; lower=2, upper=5)
        tuning = Grid(; resolution=3)
        tuned_model = TunedModel(; model=model, tuning=tuning,
                                 resampling=Holdout(; fraction_train=0.7),
                                 range=r, measure=l1)
        mach = machine(tuned_model, X, y)
        e = evaluate!(mach;
                      measure=l1,
                      resampling=Holdout(; fraction_train=0.7),
                      verbosity=0,
                      acceleration=CPU1(),
                      per_observation=false)

        @test e.measurement[1] >= 0.0 # l1 should be non-negative

        y_class = categorical(["cat", "dog", "cat", "dog", "cat", "dog"])
        model_class = CatBoostClassifier()
        r1 = range(model_class, :iterations; lower=2, upper=5)
        r2 = range(model_class, :depth; lower=1, upper=3)
        tuning_class = Grid(; resolution=2)
        tuned_model_class = TunedModel(; model=model_class, tuning=tuning_class,
                                       resampling=CV(; nfolds=2),
                                       range=[r1, r2], measure=accuracy)

        mach_class = machine(tuned_model_class, X, y_class)
        e_class = evaluate!(mach_class;
                            measure=accuracy,
                            resampling=CV(; nfolds=2),
                            verbosity=0,
                            acceleration=CPU1(),
                            per_observation=false)
        @test 0.0 <= e_class.measurement[1] <= 1.0 # accuracy between 0 and 1
    end

    @testset "generic interface tests" begin
        @testset "CatBoostRegressor" begin
            data = MLJTestInterface.make_regression()
            failures, summary = MLJTestInterface.test([CatBoostRegressor], data...;
                                                      mod=@__MODULE__, verbosity=0, # bump to debug
                                                      throw=false)
            @test isempty(failures)
        end
        @testset "CatBoostClassifier" begin
            for data in [MLJTestInterface.make_binary(),
                         MLJTestInterface.make_multiclass(),
                         MLJTestInterface.make_binary(; row_table=true),
                         MLJTestInterface.make_multiclass(; row_table=false)]
                X = data[1]
                y = data[2]
                failures, summary = MLJTestInterface.test([CatBoostClassifier], X, y;
                                                          mod=@__MODULE__, verbosity=0, # bump to debug
                                                          throw=false)
                @test isempty(failures)
            end
        end
    end
end
