"""
    idmrg2(mps, operator, alg::alg_type; verbose::Union{Bool, Integer}=true, envs = environments(mps, operator)) where {alg_type <: Union{<:IDMRG, <:IDMRG2}}
    Add some auxiliary contents to `find_groundstate` in MPSKit.jl and rename the function as `idmrg2`.
"""
function idmrg2(mps, operator, alg::alg_type; verbose::Union{Bool, Integer}=true, filename::String="default_dmrg2.jld2", envs = environments(mps, operator)) where {alg_type <: Union{<:IDMRG, <:IDMRG2}}
    (length(mps) ≤ 1 && alg isa IDMRG2) && throw(ArgumentError("unit cell should be >= 2"))
    log = alg isa IDMRG ? "IDMRG" : "IDMRG2"
    mps = copy(mps)
    iter = 0
    ϵ = calc_galerkin(mps, operator, mps, envs)
    E = zero(promote_contract(scalartype(mps), scalartype(operator)))
    E = expectation_value(mps, operator, envs)
    start_time, record_start = now(), now()
    Int(verbose) > 0 && println(log, " Sweep Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
    Int(verbose) > 0 && flush(stdout)

    state = IDMRGState(mps, operator, envs, iter, ϵ, E)
    it = IterativeSolver(alg, state)

    for (mps, envs, ϵ, ΔE) in it
        current_time = now()
        Int(verbose) > 0 && println(log, " sweep[$(it.iter)]", " ΔE = $(ΔE), ϵ = $(ϵ) | duration:", Dates.canonicalize(current_time-start_time))
        Int(verbose) > 0 && flush(stdout)
        jldopen(filename, "w") do f
            f["state"] = it.state
        end
        if ϵ ≤ alg.tol
            println(log, " task is done successfuly!")
            break
        end
        if it.iter ≥ alg.maxiter
            println(log, " task is not converged!")
            break
        end
        start_time = current_time
    end

    alg_gauge = updatetol(alg.alg_gauge, it.state.iter, it.state.ϵ)
    ψ′ = InfiniteMPS(it.state.mps.AR[1:end]; alg_gauge.tol, alg_gauge.maxiter)
    envs = recalculate!(it.state.envs, ψ′, it.state.operator, ψ′)

    jldopen(filename, "w") do f
        f["ψ"] = ψ′
        f["envs"] = envs
        f["ϵ"] = it.state.ϵ
    end

    Int(verbose) > 0 && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    return ψ′, envs, it.state.ϵ
end