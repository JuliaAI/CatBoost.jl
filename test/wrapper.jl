
@testset "Python Wrapper" begin
    using CatBoost
    @testset "`to_pandas` and `pandas_to_tbl`" begin
        tbl = Tables.columntable((floats=0.5:0.5:3.0, ints=1:6))
        pd_df = CatBoost.to_pandas(tbl)
        @test pd_df isa Py
        tbl2 = pandas_to_tbl(pd_df)
        @test tbl2 == tbl
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
