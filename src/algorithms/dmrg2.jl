"""
    dmrg2!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; alg::DMRG2=DefaultDMRG, filename::String="default_dmrg2.jld2", verbose::Union{Bool, Integer}=true, envs=environments(ψ, H))
    Add some auxiliary content to `find_groundstate` in MPSKit.jl and rename the function as `dmrg2`.
"""
function dmrg2!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; alg::DMRG2=DefaultDMRG, filename::String="default_dmrg2.jld2", verbose::Union{Bool, Integer}=true, envs=environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    trschemes =  map(d -> truncdim(d), truncdims)
    start_time, record_start = now(), now()
    Int(verbose) > 0 && println("Sweep Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
    Int(verbose) > 0 && flush(stdout)
    for iter in 1:length(truncdims)
        alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
        zerovector!(ϵs)
        errs = zeros(Float64, length(ψ))
        # left to right sweep
        for pos in 1:(length(ψ) - 1)
            @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
            Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
            _, vecs, _ = eigsolve(Hac2, ac2, 1, :SR, alg_eigsolve)
            al, c, ar, err = tsvd!(vecs[1]; p=2, trunc=trschemes[iter], alg=alg.alg_svd)
            normalize!(c)
            v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
            ϵs[pos] = abs(1 - abs(v))
            errs[pos] = err^2
            ψ.AC[pos] = (al, complex(c))
            ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            Int(verbose) > 1 && println("  SweepL2R: site $(pos) => site $(pos+1) ", Dates.format(now(), "d.u yyyy HH:MM"))
            Int(verbose) > 1 && flush(stdout)
        end

        # right to left sweep
        for pos in (length(ψ) - 2):-1:1
            @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos][-1 -2; 1] * ψ.AC[pos + 1][1 -4; -3]
            Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
            _, vecs, _ = eigsolve(Hac2, ac2, 1, :SR, alg_eigsolve)
            al, c, ar, err = tsvd!(vecs[1]; p=2, trunc=trschemes[iter], alg=alg.alg_svd)
            normalize!(c)
            v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
            ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))
            errs[pos] = max(errs[pos], err^2)
            ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            ψ.AC[pos] = (al, complex(c))
            Int(verbose) > 1 && println("  SweepR2L: site $(pos) <= site $(pos+1) ", Dates.format(now(), "d.u yyyy HH:MM"))
            Int(verbose) > 1 && flush(stdout)
        end

        err = maximum(errs)
        ϵ = maximum(ϵs)
        ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}
        D = length(ψ) <= 4 ? dim(domain(ψ[length(ψ)÷2])) : maximum([dim(domain(ψ[length(ψ)÷2])), dim(domain(ψ[length(ψ)÷2+1])), dim(domain(ψ[length(ψ)÷2-1]))])
        E₀ = expectation_value(ψ, H, envs)
        current_time = now()
        Int(verbose) > 0 && println("[$(iter)/$(length(truncdims))] sweep", " | duration:", Dates.canonicalize(current_time-start_time))
        println("  E₀ = $(E₀), D = $(D), err² = $(err), ϵ = $(ϵ)")
        flush(stdout)
        mode = (iter == 1 ? "w" : "a")
        jldopen(filename, mode) do f
            f["sweep_$(iter)_ψ"] = ψ
            f["sweep_$(iter)_ϵ"]  = ϵ
            f["sweep_$(iter)_err"] = err
            f["sweep_$(iter)_D"]  = D
        end
        start_time = current_time
    end
    record_end = now()
    Int(verbose) > 0 && println("Ended: ", Dates.format(record_end, "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(record_end-record_start))
    return ψ, envs, ϵ
end

function dmrg2(ψ, H, truncdims; kwargs...)
    return dmrg2!(copy(ψ), H, truncdims; kwargs...)
end
