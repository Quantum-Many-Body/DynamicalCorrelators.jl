"""
    dmrg2!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; alg::DMRG2=DefaultDMRG, filename::String="default_dmrg2.jld2", verbose=true, envs=environments(ψ, H))
    Add some auxiliary content to `find_groundstate` in MPSKit.jl and rename the function as `dmrg2`.
"""
function dmrg2!(ψ::AbstractFiniteMPS, H, truncdims::AbstractVector; alg::DMRG2=DefaultDMRG, filename::String="default_dmrg2.jld2", verbose=true, envs=environments(ψ, H))
    start_time, record_start = now(), now()
    verbose && println("Sweep Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
    verbose && flush(stdout)
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    trschemes =  map(d -> truncdim(d), truncdims)
    jld = isfile(filename) ? jldopen(filename, "w") : jldopen(filename, "a")
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
        end

        err = maximum(errs)
        ϵ = maximum(ϵs)
        ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}
        D = maximum([dim(domain(ψ[length(ψ)÷2])), dim(domain(ψ[length(ψ)÷2+1])), dim(domain(ψ[length(ψ)÷2-1]))])
        E₀ = expectation_value(ψ, H, envs)
        current_time = now()
        verbose && println("[$(iter)/$(length(truncdims))] sweep", " | duration:", Dates.canonicalize(current_time-start_time))
        println("  E₀ = $(E₀), D = $(D), err² = $(err), ϵ = $(ϵ)")
        flush(stdout)
        write(jld, "sweep=$(iter)_gs", ψ)
        write(jld, "sweep=$(iter)_ϵ", ϵ)
        write(jld, "sweep=$(iter)_err", err)
        write(jld, "sweep=$(iter)_D", D)
        start_time = current_time
    end
    close(jld)
    record_end = now()
    verbose && println("Ended: ", Dates.format(record_end, "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(record_end-record_start))
    return ψ, envs, ϵ
end

function dmrg2(ψ, H, truncdims; kwargs...)
    return dmrg2!(copy(ψ), H, truncdims; kwargs...)
end
