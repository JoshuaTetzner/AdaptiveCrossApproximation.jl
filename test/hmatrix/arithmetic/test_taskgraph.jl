using AdaptiveCrossApproximation
using Test

@testset "taskgraph.jl" begin
    @testset "_addtask! / _TaskGraph / _rungraph!" begin
        ACA = AdaptiveCrossApproximation
        builder = ACA._TaskGraphBuilder(2)
        ACA._addtask!(builder, (), (1,))
        ACA._addtask!(builder, (1,), (2,))
        ACA._addtask!(builder, (1,), ())
        ACA._addtask!(builder, (), (1,))
        ACA._addtask!(builder, (), (), (2,))
        @test builder.deps == [Int[], [1], [1], [1, 2, 3], [2]]
        G = ACA._TaskGraph(builder)
        for workers in (1:1, 1:3)
            finished, guard = Int[], ReentrantLock()
            @test ACA._rungraph!(G, workers) do t, _
                lock(() -> push!(finished, t), guard)
                return true
            end
            position = invperm(finished)
            @test all(position[d] < position[t] for t in 1:5 for d in builder.deps[t])
            @test !ACA._rungraph!((t, _) -> t != 1, G, workers)
            @test_throws DomainError ACA._rungraph!(
                (t, _) -> throw(DomainError(t)), G, workers
            )
        end
    end
end
