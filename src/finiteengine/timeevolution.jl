# Fast time evolution — Phase 3 of the finite engine.
#
# `fast_timestep!` mirrors MPSKit's `_timestep_finite!` / `_timestep2_finite!`
# (algorithms/timestep/tdvp.jl) step for step — same changebond! placement,
# same forward/backward integration times, same gauge placement, same edge
# cases — with the sweep running on `HalfFiniteEnvironments` instead of
# MPSKit's full-cache `FiniteEnvironments`:
#
#   1. Environments are lazy and self-invalidating (=== dependency tracking),
#      so the tensor replacements of CBE `changebond!`, of the TDVP1 gauge and
#      of the TDVP2 split automatically make the dependent environments stale —
#      no manual invalidation anywhere, identical to the DMRG fast driver.
#   2. Each environment is freed right after its last use of the sweep via the
#      same `_free_after_move!` rules as DMRG: TDVP1's extra backward C-step at
#      bond i consumes exactly the environments the one-site DMRG update
#      consumes (L2R: `C_hamiltonian(i)` uses `leftenv(i+1)`/`rightenv(i)`,
#      both alive; freeing `GRs[i+1]` afterwards matches), and TDVP2's backward
#      AC-step matches the two-site DMRG rule. Peak storage stays ~N+O(1)
#      environment tensors instead of 2(N+1).
#   3. A truncated-SVD gauge (TDVP1 with `trunc` set, e.g. CBE-TDVP) and the
#      TDVP2 split go through the block-parallel `threaded_svd_trunc` when
#      `svd_threaded` is on (`configure_finite_engine!`) — these stages are
#      otherwise serial, so there is nothing to conflict with.
#
# Like MPSKit's `timestep!` (and unlike `timestep`), the state is evolved IN
# PLACE and must already be complex: promote it once with `ψ = complex(ψ)`
# BEFORE constructing the environments — the environment array eltypes are
# fixed at construction time.

"""
    fast_timestep!(ψ, H, t, dt, alg::TDVP, envs::HalfFiniteEnvironments;
        imaginary_evolution = false, normalize = false, manual_gc = true,
        timeroutput = TimerOutputs.get_defaulttimer())
    fast_timestep!(ψ, H, t, dt, alg::TDVP2, envs::HalfFiniteEnvironments;
        imaginary_evolution = false, normalize = false, manual_gc = true,
        timeroutput = TimerOutputs.get_defaulttimer())

One full TDVP sweep (left→right→left) on the fast finite engine, evolving `ψ`
in place by `dt` starting from time `t`. Returns `(ψ, envs)`.

`manual_gc` (default `true`) runs an incremental `GC.gc(false)` after every
local update and a full collection at the end of the sweep — the same
bookkeeping as the DMRG fast driver; set it to `false` only for small systems.
`timeroutput` records the two sweeps and, inside them, the `expand` (CBE),
Hamiltonian-construction, integration and gauge/split sections; pass a shared
`TimerOutput` to accumulate across steps.
"""
function fast_timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP,
        envs::HalfFiniteEnvironments;
        imaginary_evolution::Bool = false, normalize::Bool = false,
        manual_gc::Bool = true,
        timeroutput::TimerOutput = TimerOutputs.get_defaulttimer()
    )
    scalartype(ψ) <: Complex || throw(ArgumentError(
        "fast TDVP evolves the state in place and requires a complex state; " *
        "promote it once with `ψ = complex(ψ)` before constructing the environments"
    ))
    N = length(ψ)
    # the sweep is serial, so a single allocator serves all local updates
    allocator = default_allocator(ψ, SerialScheduler())

    # sweep left to right
    @timeit timeroutput "L2R sweep" begin
        for i in 1:(N - 1)
            # 1. optionally expand the bond ahead of the local update (CBE)
            if !isnothing(alg.alg_expand)
                @timeit timeroutput "expand" changebond!(
                    i, Val(:right), ψ, H, alg.alg_expand, envs; normalize, allocator
                )
            end

            # 2. evolve the (possibly expanded) center tensor forward
            Hac = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
                i, ψ, H, ψ, envs; alg.backend, allocator
            )
            AC = @timeit timeroutput "AC_integrate" mpskit_integrate(
                Hac, ψ.AC[i], t, dt / 2, alg.integrator; imaginary_evolution
            )

            # 3. gauge: split AC -> AL[i], C[i] and move the center to i+1
            @timeit timeroutput "gauge" _fast_gauge!(
                ψ, i, Val(:right), AC, alg.alg_gauge; normalize
            )

            # 4. evolve the bond tensor backward
            Hc = @timeit timeroutput "C_hamiltonian" C_hamiltonian(
                i, ψ, H, ψ, envs; alg.backend, allocator
            )
            ψ.C[i] = @timeit timeroutput "C_integrate" mpskit_integrate(
                Hc, ψ.C[i], t + dt / 2, -dt / 2, alg.integrator; imaginary_evolution
            )

            _free_after_move!(envs, alg, Val(:right), i)
            manual_gc && GC.gc(false)
        end

        # right edge
        Hac = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
            N, ψ, H, ψ, envs; alg.backend, allocator
        )
        ψ.AC[end] = @timeit timeroutput "AC_integrate" mpskit_integrate(
            Hac, ψ.AC[end], t, dt / 2, alg.integrator; imaginary_evolution
        )
    end

    # sweep right to left
    @timeit timeroutput "R2L sweep" begin
        for i in N:-1:2
            if !isnothing(alg.alg_expand)
                @timeit timeroutput "expand" changebond!(
                    i, Val(:left), ψ, H, alg.alg_expand, envs; normalize, allocator
                )
            end

            Hac = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
                i, ψ, H, ψ, envs; alg.backend, allocator
            )
            AC = @timeit timeroutput "AC_integrate" mpskit_integrate(
                Hac, ψ.AC[i], t + dt / 2, dt / 2, alg.integrator; imaginary_evolution
            )

            @timeit timeroutput "gauge" _fast_gauge!(
                ψ, i, Val(:left), AC, alg.alg_gauge; normalize
            )

            Hc = @timeit timeroutput "C_hamiltonian" C_hamiltonian(
                i - 1, ψ, H, ψ, envs; alg.backend, allocator
            )
            ψ.C[i - 1] = @timeit timeroutput "C_integrate" mpskit_integrate(
                Hc, ψ.C[i - 1], t + dt, -dt / 2, alg.integrator; imaginary_evolution
            )

            _free_after_move!(envs, alg, Val(:left), i)
            manual_gc && GC.gc(false)
        end

        # left edge
        Hac = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
            1, ψ, H, ψ, envs; alg.backend, allocator
        )
        ψ.AC[1] = @timeit timeroutput "AC_integrate" mpskit_integrate(
            Hac, ψ.AC[1], t + dt / 2, dt / 2, alg.integrator; imaginary_evolution
        )
    end

    manual_gc && GC.gc(true)
    return ψ, envs
end

function fast_timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
        envs::HalfFiniteEnvironments;
        imaginary_evolution::Bool = false, normalize::Bool = false,
        manual_gc::Bool = true,
        timeroutput::TimerOutput = TimerOutputs.get_defaulttimer()
    )
    scalartype(ψ) <: Complex || throw(ArgumentError(
        "fast TDVP evolves the state in place and requires a complex state; " *
        "promote it once with `ψ = complex(ψ)` before constructing the environments"
    ))
    N = length(ψ)
    allocator = default_allocator(ψ, SerialScheduler())

    # sweep left to right
    @timeit timeroutput "L2R sweep" begin
        for i in 1:(N - 1)
            ac2 = _transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1])
            Hac2 = @timeit timeroutput "AC2_hamiltonian" AC2_hamiltonian(
                i, ψ, H, ψ, envs; alg.backend, allocator
            )
            ac2′ = @timeit timeroutput "AC2_integrate" mpskit_integrate(
                Hac2, ac2, t, dt / 2, alg.integrator; imaginary_evolution
            )

            nal, nc, nar = @timeit timeroutput "split" _tdvp2_split(ac2′, alg)
            normalize && normalize!(nc)
            ψ.AC[i] = (nal, complex(nc))
            ψ.AC[i + 1] = (complex(nc), _transpose_front(nar))

            if i != N - 1
                Hac = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
                    i + 1, ψ, H, ψ, envs; alg.backend, allocator
                )
                ψ.AC[i + 1] = @timeit timeroutput "AC_integrate" mpskit_integrate(
                    Hac, ψ.AC[i + 1], t + dt / 2, -dt / 2, alg.integrator;
                    imaginary_evolution
                )
            end

            _free_after_move!(envs, alg, Val(:right), i)
            manual_gc && GC.gc(false)
        end
    end

    # sweep right to left
    @timeit timeroutput "R2L sweep" begin
        for i in N:-1:2
            ac2 = _transpose_front(ψ.AL[i - 1]) * _transpose_tail(ψ.AC[i])
            Hac2 = @timeit timeroutput "AC2_hamiltonian" AC2_hamiltonian(
                i - 1, ψ, H, ψ, envs; alg.backend, allocator
            )
            ac2′ = @timeit timeroutput "AC2_integrate" mpskit_integrate(
                Hac2, ac2, t + dt / 2, dt / 2, alg.integrator; imaginary_evolution
            )

            nal, nc, nar = @timeit timeroutput "split" _tdvp2_split(ac2′, alg)
            normalize && normalize!(nc)
            ψ.AC[i - 1] = (nal, complex(nc))
            ψ.AC[i] = (complex(nc), _transpose_front(nar))

            if i != 2
                Hac = @timeit timeroutput "AC_hamiltonian" AC_hamiltonian(
                    i - 1, ψ, H, ψ, envs; alg.backend, allocator
                )
                ψ.AC[i - 1] = @timeit timeroutput "AC_integrate" mpskit_integrate(
                    Hac, ψ.AC[i - 1], t + dt, -dt / 2, alg.integrator;
                    imaginary_evolution
                )
            end

            _free_after_move!(envs, alg, Val(:left), i - 1)
            manual_gc && GC.gc(false)
        end
    end

    manual_gc && GC.gc(true)
    return ψ, envs
end

# TDVP2 split of the evolved two-site center: block-parallel when enabled via
# `configure_finite_engine!(; svd_threaded = true)`, MPSKit's in-place
# `svd_trunc!` otherwise (`ac2′` is a fresh tensor returned by the integrator,
# so mutating it is safe)
function _tdvp2_split(ac2′, alg::TDVP2)
    if _SVD_THREADED[] && Threads.nthreads() > 1
        nal, nc, nar, = threaded_svd_trunc(
            ac2′, TruncatedAlgorithm(alg.alg_svd, alg.trunc)
        )
    else
        nal, nc, nar = svd_trunc!(ac2′; trunc = alg.trunc, alg = alg.alg_svd)
    end
    return nal, nc, nar
end
