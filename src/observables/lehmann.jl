# Low-lying-state removal for zero-temperature dynamical correlators
# (Paeckel et al., Ann. Phys. 411, 167998 (2019), Sec. 7.3).
#
# The charged state ket = op·gs is split into the subspace spanned by a set of
# precomputed low-lying eigenstates of the charged sector and a remainder:
#
#     ket = Σₙ cₙ ψₙ + rest,        cₙ = ⟨ψₙ|ket⟩ / ⟨ψₙ|ψₙ⟩.
#
# Only the remainder is time-evolved with MPS/TDVP; the removed components are
# propagated analytically through their eigen-energies (the Lehmann part)
#
#     gf_low[i, k] = -im Σₙ cₙ dₙ[i] exp(-im ΔEₙ tₖ)                (id ≤ L)
#     gf_low[i, k] = -im Σₙ conj(cₙ) conj(dₙ[i]) exp(+im ΔEₙ tₖ)    (id > L)
#
# with dₙ = sweep_dot(gs, op, ψₙ) and ΔEₙ = Eₙ - E₀, and added back slice by
# slice. The excited states are meant to come from `excited_states` run in the
# charge sector reached by `op`: the creation-operator calls (id ≤ L) and the
# annihilation-operator calls (id > L) each take their own set of states.

function _median(x::AbstractVector{<:Real})
    y = sort!(collect(x))
    n = length(y)
    n == 0 && throw(ArgumentError("cannot take the median of an empty collection"))
    mid = (n + 1) ÷ 2
    return isodd(n) ? y[mid] : (y[mid] + y[mid + 1]) / 2
end

# Recommendation table. The evolution length is governed by the
# *within-sector* excitation energies εₙ = Eₙ - E₀ˢᵉᶜ, where E₀ˢᵉᶜ is the
# ground-state energy of the charged sector (the lowest computed state, i.e.
# the threshold pole): the threshold pole itself only contributes the
# analytically known carrier phase exp(-im ω₁ t), so the slowest component
# the evolution must resolve after removing the first k states is εₖ₊₁.
# This is the review's t ~ 1/E_{m+1} criterion with the energy zero at the
# sector ground state. Row k (nremove = k = 1..m-1) is the definite window
# t ~ (1/εₖ₊₁, 2π/εₖ₊₁) (minimal phase accumulation ~ one full period); only
# when *all* m states are removed is ε_{m+1} unknown and estimated by
# εₘ + δ̂ with δ̂ the median pole spacing.
function _recommend_tmax_table(dEs::AbstractVector{<:Real}, eta)
    p = sort!(collect(Float64, dEs))
    m = length(p)
    m == 0 && throw(ArgumentError("no excitation energies given"))
    within = p .- p[1]
    spacings = diff(p)
    positive = filter(s -> s > 0, spacings)  # degenerate poles carry no resolution cost
    δ = !isempty(positive) ? _median(positive) : nothing
    rows = [(nremove=k, slowest=within[k + 1]) for k in 1:(m - 1)]
    estimate = if m >= 2 && !isnothing(δ) && δ > 0
        (; next=within[m] + δ)
    else
        nothing
    end
    return (; positions=p, within=within, spacings=spacings, δ=δ,
              rows=rows, estimate=estimate, eta=eta)
end

# print a row's reference window: t1 = 1/ε (accumulate O(1) phase, minimal
# criterion) to t2π = 2π/ε (one full period of the slowest component)
_print_tmax_pair(io::IO, ε) = @printf(io, "(%8.4f ~ %8.4f)", 1 / ε, 2π / ε)

function _print_tmax_table(rec, E0)
    m = length(rec.positions)
    @printf("recommend_tmax (reference: sector ground state | E₀(N) = %.6f)\n", E0)
    print("  pole positions ωₙ = Eₙ - E₀(N):  ")
    foreach(x -> @printf("%.6f  ", x), rec.positions)
    print("\n  within-sector εₙ = Eₙ - E₀ˢᵉᶜ:   ")
    foreach(x -> @printf("%.6f  ", x), rec.within)
    if !isempty(rec.spacings)
        print("\n  spacings δₙ = ωₙ₊₁ - ωₙ:         ")
        foreach(x -> @printf("%.6f  ", x), rec.spacings)
    end
    println("\n  " * "-"^72)
    println("  tmax window: (1/ε ~ 2π/ε) = minimal phase accumulation ~ one full period")
    println("  use 0 states (no removal)    no energy-based bound: the sharp threshold pole stays")
    println("                               in the signal; tmax is set by the broadening (see eta)")
    for row in rec.rows
        label = "use first $(row.nremove) state(s)"
        if row.slowest > 0
            @printf("  %-29s tmax ≳ ", label)
            _print_tmax_pair(stdout, row.slowest)
            @printf("  (slowest: ε%d = %.6f)\n", row.nremove + 1, row.slowest)
        else
            @printf("  %-29s degenerate with the sector ground state — no gain until the whole multiplet is removed\n", label)
        end
    end
    if !isnothing(rec.estimate)
        @printf("  %-29s tmax ≳ ", "use all $m state(s)")
        _print_tmax_pair(stdout, rec.estimate.next)
        @printf("  (estimate: ε%d ≈ %.6f, δ̂ = %.6f)\n", m + 1, rec.estimate.next, rec.δ)
    else
        @printf("  %-29s no estimate (need ≥ 2 states with distinct energies)\n", "use all $m state(s)")
    end
    if !isnothing(rec.eta)
        @printf("  eta = %.4g: resolution floor tmax ≳ 2π/η = %.4f (Gaussian window support ≈ 3/η = %.4f)\n", rec.eta, 2π / rec.eta, 3 / rec.eta)
        if !isnothing(rec.δ)
            if rec.δ > rec.eta
                @printf("  removal depth: δ̂ = %.4f > η = %.4g — the first uncomputed pole is likely an isolated sharp feature; consider computing more excited states\n", rec.δ, rec.eta)
            else
                @printf("  removal depth: δ̂ = %.4f < η = %.4g — the remainder is smooth at this resolution; the removal set is deep enough\n", rec.δ, rec.eta)
            end
        end
    end
    println("  note: entries assume the corresponding state actually couples to the source (check |cₙ|² from `lehmann_report` or the dcorrelator pre-check)")
    flush(stdout)
    return nothing
end

"""
    recommend_tmax(energies, E0; eta = nothing, verbose = true) -> NamedTuple

Recommend the time-evolution length for `dcorrelator` with low-lying removal
(Sec. 7.3 of Paeckel et al.). `energies` are the absolute energies of the
computed low-lying states (as saved by `excited_states`) and `E0` the
ground-state energy of the unperturbed sector.

The relevant energy zero is the *sector* ground state (the lowest computed
state): the threshold pole only contributes the analytically known carrier
phase, so the slowest component the time evolution must resolve after
removing the first k states is the within-sector excitation energy
εₖ₊₁ = Eₖ₊₁ - E₀ˢᵉᶜ — the review's `t ~ 1/E_{m+1}` criterion with the
reference at the sector ground state. The printed table gives a definite
window `(1/εₖ₊₁ ~ 2π/εₖ₊₁)` for every `k = 1, …, m - 1` (`rows`), where the
lower end is the minimal criterion (the slowest component accumulates O(1)
phase) and the upper end asks for one full period; with all m states
removed, ε_{m+1} is unknown and estimated as εₘ + δ̂ with δ̂ the median pole
spacing (`estimate`). With no removal there is no energy-based bound: the
sharp threshold pole stays in the signal and the required length is set by
the broadening alone.

With `eta` (the broadening used e.g. in `fourier_rw`) two extra checks are
printed: the resolution floor `tmax ≳ 2π/η`, and a removal-depth verdict
comparing the estimated spacing δ̂ of the first uncomputed pole against η
(δ̂ > η means a likely isolated sharp feature remains — compute more states).

Note that every entry assumes the corresponding state actually couples to the
source; a state with vanishing |cₙ|² (see `lehmann_report`) does not force a
long evolution.

Returns `(; positions, within, spacings, δ, rows, estimate, eta)`.
"""
function recommend_tmax(energies::AbstractVector{<:Real}, E0::Real;
                        eta::Union{Nothing,Real}=nothing, verbose::Bool=true)
    rec = _recommend_tmax_table(Float64.(energies .- E0), eta)
    verbose && _print_tmax_table(rec, E0)
    return rec
end

# internal warning used inside dcorrelator when tmax_check = true; returns the
# recommendation NamedTuple (or nothing for non-real spectra, e.g. non-Hermitian H)
function _lehmann_tmax_warn(dEs, tmax_user, eta)
    any(e -> !(e isa Real), dEs) && return nothing
    rec = _recommend_tmax_table(Float64.(dEs), eta)
    est = rec.estimate
    if !isnothing(est)
        t1, t2π = 1 / est.next, 2π / est.next
        if tmax_user < t1
            @warn "times max = $tmax_user is below the recommended window ($t1 ~ $t2π) for the full removal set (slowest remainder component ε ≈ $(est.next), sector-referenced criterion): the remainder is under-resolved; inspect with `recommend_tmax`, disable with `tmax_check = false`"
        elseif tmax_user < t2π
            @warn "times max = $tmax_user covers the minimal criterion but is below one full period of the slowest remainder component (recommended window ($t1 ~ $t2π), ε ≈ $(est.next)); inspect with `recommend_tmax`, disable with `tmax_check = false`"
        end
    end
    if !isnothing(eta) && tmax_user < 2π / eta
        @warn "times max = $tmax_user is below the resolution floor 2π/η = $(2π / eta) for eta = $eta: the Fourier resolution is coarser than the broadening (disable with `tmax_check = false`)"
    end
    return rec
end
# MPSKit's three-argument dot requires bra and ket to share one tensor type;
# promote to complex for mixed real/complex input
function _lehmann_dot3(bra, O, ket)
    try
        return dot(bra, O, ket)
    catch err
        err isa MethodError || rethrow()
        return dot(complex(bra), O, complex(ket))
    end
end

function _lehmann_energies(H, excited, energies)
    if !isnothing(energies)
        length(energies) == length(excited) || throw(ArgumentError("`energies` has length $(length(energies)) but $(length(excited)) excited states were given"))
        return collect(energies)
    end
    return [expectation_value(ψₙ, H) for ψₙ in excited]
end

function _lehmann_orthocheck(excited, orth_tol; verbose=false)
    m = length(excited)
    S = Matrix{ComplexF64}(undef, m, m)
    for i in 1:m
        for j in i:m
            S[i, j] = dot(excited[i], excited[j])
            S[j, i] = conj(S[i, j])
        end
    end
    maxoff = 0.0
    maxdev = 0.0
    for i in 1:m
        maxdev = max(maxdev, abs(S[i, i] - 1))
        for j in (i + 1):m
            maxoff = max(maxoff, abs(S[i, j]))
        end
    end
    maxoff > orth_tol && throw(ArgumentError("the given excited states are not orthonormal: max off-diagonal overlap $(maxoff) > orth_tol = $(orth_tol); re-run `excited_states` (its report prints the overlap matrix) so that the Lehmann projection formula stays valid"))
    verbose && @printf("  orthonormality: max off-diagonal = %.3e | max |⟨ψₙ|ψₙ⟩-1| = %.3e\n", maxoff, maxdev)
    return S
end

# exact overlaps cₙ = ⟨ψₙ|chargedMPO(op,idx)|gs⟩ / ⟨ψₙ|ψₙ⟩ for every source,
# from MPSKit's three-argument dot — no chargedMPS compression needed
function _lehmann_exact_overlaps(gs, op, ids, excited)
    L = length(gs)
    m = length(excited)
    cs = Matrix{ComplexF64}(undef, m, length(ids))
    for (d, id) in enumerate(ids)
        idx = _dcorrelator_site(id, L)
        cmpo = chargedMPO(op, idx, L)
        for n in 1:m
            ψₙ = excited[n]
            c = try
                _lehmann_dot3(ψₙ, cmpo, gs)
            catch err
                if err isa SpaceMismatch
                    throw(ArgumentError("excited state $n has a quantum-number sector incompatible with `op` acting on `gs` (source id = $id): run `excited_states` in the charge sector reached by the operator"))
                end
                rethrow()
            end
            cs[n, d] = c / dot(ψₙ, ψₙ)
        end
    end
    return cs
end

# print the per-source overlap table and apply the two-level policy:
# throw when *every* state is invisible to a source, warn when at least
# `overlap_warn_fraction` of them are (the user decides whether to abort)
function _lehmann_overlap_check(cs, ids, dEs; overlap_abstol, overlap_warn_fraction, verbose)
    m = size(cs, 1)
    if verbose
        print("  excitation energies ΔEₙ = [")
        for n in 1:m
            @printf("%.6f%s", real(dEs[n]), n == m ? "" : ", ")
        end
        print("]\n")
    end
    for (d, id) in enumerate(ids)
        w = abs2.(cs[:, d])
        nbelow = count(<(overlap_abstol), w)
        if verbose
            @printf("  id = %d: captured Σ|cₙ|² = %.3e | below tol: %d/%d | |cₙ|² = [", id, sum(w), nbelow, m)
            for n in 1:m
                @printf("%.2e%s", w[n], n == m ? "" : ", ")
            end
            print("]\n")
            flush(stdout)
        end
        if nbelow == m
            throw(ArgumentError("source id = $id: all $m excited states have |cₙ|² < overlap_abstol = $overlap_abstol, so the removal cannot help this source; compute more excited states in this charge sector (or check that the sector is correct)"))
        elseif nbelow / m >= overlap_warn_fraction
            @warn "source id = $id: $nbelow of $m excited states have |cₙ|² < $overlap_abstol — most computed states are invisible to this source; the removal gain may be small (inspect the printed weights and decide whether to continue or compute more excited states)"
        end
    end
    return nothing
end

# remainder ket − Σₙ cₙψₙ as an MPS. MPSKit's `+` builds the exact direct-sum
# state (isometric embedding into the direct sum of virtual spaces, so no
# cancellation is lost); each addition is followed by a one-shot zip-up
# compression through the identity MPO to keep the bond dimension in check
function _lehmann_rest(ket, cs, excited, H, rest_trunc, compress_rest)
    rest = ket
    IM = identityMPO(H)
    alg = Zipup(; trunc=rest_trunc)
    for n in eachindex(excited)
        raw = rest + (-cs[n]) * excited[n]
        rest = compress_rest ? first(approximate((IM, raw), alg)) : raw
    end
    return rest
end

# gf_low[i, r] of a single source on the recorded time grid
function _lehmann_gf_low(cs, ds, dEs, times, record_indices, id, L)
    low = zeros(ComplexF64, L, length(record_indices))
    for (r, k) in enumerate(record_indices)
        t = times[k]
        if id <= L
            low[:, r] = -im .* (ds * (cs .* exp.(-im .* dEs .* t)))
        else
            low[:, r] = -im .* conj.(ds * (cs .* exp.(im .* dEs .* t)))
        end
    end
    return low
end

# Gorkov version: the raw sweep_dot contributions of the removed states are
# Σₙ cₙ dₙ exp(-im Eₙ t) with *absolute* energies; the shared helper then
# applies the same phase/conjugation conventions as for the remainder data
function _lehmann_gf_low_gorkov(cs, ds, ds_anom, dEs, E0, times, record_indices, id, L)
    nt = length(record_indices)
    G11 = zeros(ComplexF64, L, nt)
    G12 = similar(G11)
    G21 = similar(G11)
    G22 = similar(G11)
    Eabs = dEs .+ E0
    for (r, k) in enumerate(record_indices)
        t = times[k]
        factors = exp.(-im .* Eabs .* t)
        sd_low = ds * (cs .* factors)
        sd_low_anom = ds_anom * (cs .* factors)
        G11[:, r], G12[:, r], G21[:, r], G22[:, r] =
            _dcorrelator_gorkov_vectors(id, L, E0, t, sd_low, sd_low_anom)
    end
    return (; G11, G12, G21, G22)
end

"""
    lehmann_eval(cs, ds, dEs, times, id, L) -> Matrix{ComplexF64}

Evaluate the removed low-lying (Lehmann) part of a zero-temperature
`dcorrelator` on an arbitrary time grid. `cs`, `ds` and `dEs` are the overlap
coefficients, the measurement vectors and the excitation energies, either from
the NamedTuple returned by `dcorrelator(..., excited)` or from the
`"lowlying_cs" / "lowlying_ds" / "lowlying_dE"` entries of its output file.

The intended use after linear prediction: extrapolate `gf_rest` onto a longer
grid, re-evaluate the Lehmann part on the same grid with this function, and
add the two before the Fourier transform. `id` and `L` follow the usual
doubled convention (`id ≤ L` greater part, `id > L` lesser part).
"""
function lehmann_eval(cs::AbstractVector, ds::AbstractMatrix, dEs::AbstractVector, times, id::Integer, L::Integer)
    return _lehmann_gf_low(cs, ds, dEs, times, eachindex(times), id, L)
end

# save a pure-Lehmann result (remainder below skip_rest_tol) in the same
# format as the regular dcorrelator output
function _lehmann_save_skip(gf_path, times, record_indices, id, gf_low, lowlying)
    !isdir(gf_path) && mkpath(gf_path)
    filename = joinpath(gf_path, "gf_start=$(times[record_indices[1]])_end=$(times[record_indices[end]])_id=$(id).jld2")
    jldopen(filename, "w") do f
        f["times"] = times
        f["record_indices"] = record_indices
        f["id"] = id
        for r in eachindex(record_indices)
            f["pro_$(r)"] = gf_low[:, r]
        end
        for (k, v) in pairs(lowlying)
            f["lowlying_$(k)"] = v
        end
    end
    return nothing
end

function _lehmann_save_skip_gorkov(gf_path, times, record_indices, id, gf_low, lowlying)
    !isdir(gf_path) && mkpath(gf_path)
    filename = _dcorrelator_gorkov_filename(gf_path, times[record_indices[1]], times[record_indices[end]], id)
    jldopen(filename, "w") do f
        f["times"] = times
        f["record_indices"] = record_indices
        f["id"] = id
        f["gorkov_keys"] = string.(_DCORRELATOR_GORKOV_KEYS)
        for (k, v) in pairs(lowlying)
            f["lowlying_$(k)"] = v
        end
    end
    for r in eachindex(record_indices)
        _dcorrelator_save_gorkov_slice!(filename, r, gf_low.G11[:, r], gf_low.G12[:, r], gf_low.G21[:, r], gf_low.G22[:, r])
    end
    return nothing
end

# shared per-source worker: build the charged ket, project out the excited
# subspace, evolve the remainder through the regular `dcorrelator` method
# (keyword `source`), and add the analytic Lehmann part (keyword `gf_low`)
function _lehmann_dcorrelator_one(gs, H, op, id, excited, E0, dEs, ds, ds_anom;
                                  verbose, save, gf_path, disk, times, record_indices, n,
                                  anomalous, approxalg, tdvp1, tdvp2,
                                  compress_rest, rest_trunc, skip_rest_tol)
    L = length(H)
    idx = _dcorrelator_site(id, L)
    record_indices, record_first, record_last = _dcorrelator_record_window(times, record_indices)
    timer = TimerOutput()
    ket = @timeit timer "setup / chargedMPS" chargedMPS(op, gs, idx, approxalg; disk)
    cs = [dot(ψₙ, ket) / dot(ψₙ, ψₙ) for ψₙ in excited]
    ket_weight = abs2(norm(ket))
    rest = @timeit timer "setup / remainder" _lehmann_rest(ket, cs, excited, H, rest_trunc, compress_rest)
    rest_weight = abs2(norm(rest)) / ket_weight
    verbose && @printf("  id = %d: remainder weight ‖rest‖²/‖ket‖² = %.3e\n", id, rest_weight)
    verbose && flush(stdout)
    gf_low = if isnothing(anomalous)
        _lehmann_gf_low(cs, ds, dEs, times, record_indices, id, L)
    else
        _lehmann_gf_low_gorkov(cs, ds, ds_anom, dEs, E0, times, record_indices, id, L)
    end
    lowlying = if isnothing(anomalous)
        (; cs=cs, ds=ds, dE=dEs)
    else
        (; cs=cs, ds=ds, ds_anom=ds_anom, dE=dEs)
    end
    if rest_weight < skip_rest_tol
        verbose && println("  id = $id: remainder weight below skip_rest_tol = $skip_rest_tol; skipping time evolution (pure Lehmann result)")
        verbose && flush(stdout)
        if isnothing(anomalous)
            save && _lehmann_save_skip(gf_path, times, record_indices, id, gf_low, lowlying)
            return (; gf=gf_low, gf_low=gf_low, gf_rest=zero(gf_low), cs=cs, ds=ds, dE=dEs, E0=E0)
        else
            save && _lehmann_save_skip_gorkov(gf_path, times, record_indices, id, gf_low, lowlying)
            return (; gf=gf_low, gf_low=gf_low, gf_rest=map(zero, gf_low), cs=cs, ds=ds, ds_anom=ds_anom, dE=dEs, E0=E0)
        end
    end
    gf = dcorrelator(gs, H, op, id;
                     source=rest, gf_low=gf_low, lowlying=lowlying,
                     verbose=verbose, save=save, gf_path=gf_path, disk=disk,
                     times=times, record_indices=record_indices, n=n,
                     anomalous=anomalous, approxalg=approxalg, tdvp1=tdvp1, tdvp2=tdvp2)
    if isnothing(anomalous)
        return (; gf=gf, gf_low=gf_low, gf_rest=gf - gf_low, cs=cs, ds=ds, dE=dEs, E0=E0)
    else
        return (; gf=gf, gf_low=gf_low, gf_rest=map(-, gf, gf_low), cs=cs, ds=ds, ds_anom=ds_anom, dE=dEs, E0=E0)
    end
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap,
                id::Integer, excited::AbstractVector{<:FiniteMPS}; kwargs...)

Zero-temperature dynamical correlation from one source site with removal of
low-lying states (Paeckel et al., Ann. Phys. 411, 167998 (2019), Sec. 7.3).

The charged state `ket = chargedMPO(op, idx) * gs` is decomposed into the
subspace spanned by the precomputed eigenstates `excited` of the charged
sector and a remainder: `ket = Σₙ cₙ ψₙ + rest`. Only `rest` is time-evolved
with MPS/TDVP; the removed components are propagated analytically through
their eigen-energies (Lehmann part) and added back slice by slice, so the
returned total correlator is identical in exact arithmetic — but the evolved
part no longer contains the slowest oscillations. Since `excited` must live in
the sector reached by `op`, creation-operator runs (`id ≤ L`) and
annihilation-operator runs (`id > L`) each take their own set of states,
typically from separate `excited_states` runs saved to disk.

# Additional keywords
- `energies = nothing`: precomputed energies of `excited`; computed from
  `expectation_value(ψₙ, H)` when omitted.
- `overlap_abstol = 1e-10`: threshold on |cₙ|² for the overlap pre-check. The
  exact overlaps are computed upfront through MPSKit's three-argument `dot`
  (no chargedMPS compression) and printed. If *every* state is below the
  threshold for the source, an `ArgumentError` is thrown before anything
  expensive happens — compute more excited states.
- `overlap_warn_fraction = 0.5`: warn (without aborting) when at least this
  fraction of the states is below `overlap_abstol`.
- `orth_tol = 1e-6`: the input states are checked for orthonormality, which
  the projection formula relies on.
- `compress_rest = true`, `rest_trunc = truncerror(; rtol=1e-10)`: the
  remainder is built by exact direct sums followed by zip-up compression with
  this truncation (pass e.g. `truncrank(D)` to cap the bond dimension).
- `skip_rest_tol = 1e-12`: if ‖rest‖²/‖ket‖² falls below this, the time
  evolution is skipped and the pure Lehmann result returned.
- `tmax_check = true`, `eta = nothing`: warn when `times`
  max is below the recommended evolution window (see [`recommend_tmax`](@ref));
  passing the broadening `eta` additionally enables the `2π/η` resolution-floor
  warning.

Returns a NamedTuple `(; gf, gf_low, gf_rest, cs, ds, dE, E0)` with the total,
the analytic Lehmann part and the evolved remainder part (apply linear
prediction to `gf_rest` only, then re-add `lehmann_eval` on the extended
grid). Saved files keep the regular format with the total in `"pro_r"`, plus
`"lowlying_cs" / "lowlying_ds" / "lowlying_dE"` entries. With `anomalous`,
`gf`, `gf_low` and `gf_rest` are NamedTuples `(; G11, G12, G21, G22)`.
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, id::Integer,
                     excited::AbstractVector{<:FiniteMPS};
                     verbose=true,
                     save::Bool=true,
                     gf_path::String="./",
                     disk::Union{Bool,AbstractString}=false,
                     times::AbstractRange=0:0.05:5.0,
                     record_indices::AbstractArray=1:length(times),
                     n::Integer=3,
                     anomalous::Union{Nothing,AbstractTensorMap}=nothing,
                     approxalg=myDMRG2(),
                     tdvp1=myTDVP1(),
                     tdvp2=myTDVP2(; trunc=truncerror(; rtol=1e-3)),
                     energies::Union{Nothing,AbstractVector}=nothing,
                     overlap_abstol::Float64=1e-10,
                     overlap_warn_fraction::Float64=0.5,
                     orth_tol::Float64=1e-6,
                     compress_rest::Bool=true,
                     rest_trunc=truncerror(; rtol=1e-10),
                     skip_rest_tol::Float64=1e-12,
                     tmax_check::Bool=true,
                     eta::Union{Nothing,Real}=nothing)
    L = length(H)
    1 <= id <= 2L || throw(ArgumentError("id must be inside 1:$(2L)"))
    record_indices, record_first, record_last = _dcorrelator_record_window(times, record_indices)
    verbose && println("dcorrelator / low-lying removal: $(length(excited)) excited state(s), source id = $id")
    Es = _lehmann_energies(H, excited, energies)
    E0 = expectation_value(gs, H)
    dEs = Es .- E0
    _lehmann_orthocheck(excited, orth_tol; verbose=verbose)
    tmax_check && _lehmann_tmax_warn(dEs, times[record_last], eta)
    cs_exact = _lehmann_exact_overlaps(gs, op, (id,), excited)
    _lehmann_overlap_check(cs_exact, (id,), dEs; overlap_abstol, overlap_warn_fraction, verbose=verbose)
    ds = reduce(hcat, (sweep_dot(gs, op, ψₙ) for ψₙ in excited))
    ds_anom = isnothing(anomalous) ? nothing : reduce(hcat, (sweep_dot(gs, anomalous, ψₙ) for ψₙ in excited))
    return _lehmann_dcorrelator_one(gs, H, op, id, excited, E0, dEs, ds, ds_anom;
                                    verbose, save, gf_path, disk, times, record_indices, n,
                                    anomalous, approxalg, tdvp1, tdvp2,
                                    compress_rest, rest_trunc, skip_rest_tol)
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap,
                indices::AbstractArray, excited::AbstractVector{<:FiniteMPS}; kwargs...)

Multi-source version of the low-lying-removal `dcorrelator`: the source
channels in `indices` are evaluated in parallel with Distributed and stored in
a `SharedArray`. The same `excited` states are reused for every source — the
overlap pre-check runs for all sources on the main process first (failing
fast, before any time evolution), and the source-independent measurement
vectors `dₙ = sweep_dot(gs, op, ψₙ)` are computed only once. Keywords and the
returned NamedTuple follow the single-source method, with `gf`, `gf_low`,
`gf_rest` of size `(length(H), length(indices), length(record_indices))` and
`cs` of size `(length(excited), length(indices))`.
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray,
                     excited::AbstractVector{<:FiniteMPS};
                     verbose=true,
                     save::Bool=true,
                     gf_path::String="./",
                     disk::Union{Bool,AbstractString}=false,
                     times::AbstractRange=0:0.05:5.0,
                     record_indices::AbstractArray=1:length(times),
                     n::Integer=3,
                     anomalous::Union{Nothing,AbstractTensorMap}=nothing,
                     approxalg=myDMRG2(),
                     tdvp1=myTDVP1(),
                     tdvp2=myTDVP2(; trunc=truncerror(; rtol=1e-3)),
                     energies::Union{Nothing,AbstractVector}=nothing,
                     overlap_abstol::Float64=1e-10,
                     overlap_warn_fraction::Float64=0.5,
                     orth_tol::Float64=1e-6,
                     compress_rest::Bool=true,
                     rest_trunc=truncerror(; rtol=1e-10),
                     skip_rest_tol::Float64=1e-12,
                     tmax_check::Bool=true,
                     eta::Union{Nothing,Real}=nothing)
    L = length(H)
    ids = collect(indices)
    all(x -> 1 <= x <= 2L, ids) || throw(ArgumentError("indices must be inside 1:$(2L)"))
    record_indices, record_first, record_last = _dcorrelator_record_window(times, record_indices)
    nids = length(ids)
    nt = length(record_indices)
    m = length(excited)
    verbose && println("dcorrelator / low-lying removal: $m excited state(s), $nids source channel(s)")
    Es = _lehmann_energies(H, excited, energies)
    E0 = expectation_value(gs, H)
    dEs = Es .- E0
    _lehmann_orthocheck(excited, orth_tol; verbose=verbose)
    tmax_check && _lehmann_tmax_warn(dEs, times[record_last], eta)
    cs_exact = _lehmann_exact_overlaps(gs, op, ids, excited)
    _lehmann_overlap_check(cs_exact, ids, dEs; overlap_abstol, overlap_warn_fraction, verbose=verbose)
    ds = reduce(hcat, (sweep_dot(gs, op, ψₙ) for ψₙ in excited))
    ds_anom = isnothing(anomalous) ? nothing : reduce(hcat, (sweep_dot(gs, anomalous, ψₙ) for ψₙ in excited))
    cs_all = SharedArray{ComplexF64, 2}(m, nids)
    if isnothing(anomalous)
        gf = SharedArray{ComplexF64, 3}(L, nids, nt)
        gf_low = SharedArray{ComplexF64, 3}(L, nids, nt)
        G11 = G12 = G21 = G22 = nothing
        L11 = L12 = L21 = L22 = nothing
    else
        gf = gf_low = nothing
        G11 = SharedArray{ComplexF64, 3}(L, nids, nt)
        G12 = SharedArray{ComplexF64, 3}(L, nids, nt)
        G21 = SharedArray{ComplexF64, 3}(L, nids, nt)
        G22 = SharedArray{ComplexF64, 3}(L, nids, nt)
        L11 = SharedArray{ComplexF64, 3}(L, nids, nt)
        L12 = SharedArray{ComplexF64, 3}(L, nids, nt)
        L21 = SharedArray{ComplexF64, 3}(L, nids, nt)
        L22 = SharedArray{ComplexF64, 3}(L, nids, nt)
    end
    @sync @distributed for d in eachindex(ids)
        out = _lehmann_dcorrelator_one(gs, H, op, ids[d], excited, E0, dEs, ds, ds_anom;
                                       verbose, save, gf_path, disk, times, record_indices, n,
                                       anomalous, approxalg, tdvp1, tdvp2,
                                       compress_rest, rest_trunc, skip_rest_tol)
        cs_all[:, d] = out.cs
        if isnothing(anomalous)
            gf[:, d, :] = out.gf
            gf_low[:, d, :] = out.gf_low
        else
            G11[:, d, :] = out.gf.G11
            G12[:, d, :] = out.gf.G12
            G21[:, d, :] = out.gf.G21
            G22[:, d, :] = out.gf.G22
            L11[:, d, :] = out.gf_low.G11
            L12[:, d, :] = out.gf_low.G12
            L21[:, d, :] = out.gf_low.G21
            L22[:, d, :] = out.gf_low.G22
        end
    end
    csa = zeros(ComplexF64, m, nids)
    csa .= cs_all
    if isnothing(anomalous)
        gfs = zeros(ComplexF64, L, nids, nt)
        gfs .= gf
        lows = zeros(ComplexF64, L, nids, nt)
        lows .= gf_low
        return (; gf=gfs, gf_low=lows, gf_rest=gfs .- lows, cs=csa, ds=ds, dE=dEs, E0=E0)
    else
        a11 = zeros(ComplexF64, L, nids, nt); a11 .= G11
        a12 = zeros(ComplexF64, L, nids, nt); a12 .= G12
        a21 = zeros(ComplexF64, L, nids, nt); a21 .= G21
        a22 = zeros(ComplexF64, L, nids, nt); a22 .= G22
        b11 = zeros(ComplexF64, L, nids, nt); b11 .= L11
        b12 = zeros(ComplexF64, L, nids, nt); b12 .= L12
        b21 = zeros(ComplexF64, L, nids, nt); b21 .= L21
        b22 = zeros(ComplexF64, L, nids, nt); b22 .= L22
        gfs = (; G11=a11, G12=a12, G21=a21, G22=a22)
        lows = (; G11=b11, G12=b12, G21=b21, G22=b22)
        return (; gf=gfs, gf_low=lows, gf_rest=map(-, gfs, lows), cs=csa, ds=ds, ds_anom=ds_anom, dE=dEs, E0=E0)
    end
end

"""
    lehmann_report(gs, H, op, ids, excited; kwargs...) -> NamedTuple

Pre-flight diagnostic for `dcorrelator` with low-lying removal. For every
source in `ids` (an Integer or a collection) it computes the exact overlaps
|cₙ|² = |⟨ψₙ|op·gs⟩|² of the excited states through MPSKit's three-argument
`dot` — without any chargedMPS compression or time evolution — prints them,
checks the orthonormality of `excited`, and prints the recommended
time-evolution window from `recommend_tmax`. Run this before the expensive
calculations to decide whether the computed excited states suffice.

Returns `(; cs, dE, S, tmax)` with the overlap matrix (states × sources), the
excitation energies, the overlap matrix of the excited states and the
`recommend_tmax` result (`nothing` for non-real spectra).
"""
function lehmann_report(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap,
                        ids, excited::AbstractVector{<:FiniteMPS};
                        energies::Union{Nothing,AbstractVector}=nothing,
                        overlap_abstol::Float64=1e-10,
                        overlap_warn_fraction::Float64=0.5,
                        orth_tol::Float64=1e-6,
                        eta::Union{Nothing,Real}=nothing,
                        verbose::Bool=true)
    idv = ids isa Integer ? [ids] : collect(ids)
    Es = _lehmann_energies(H, excited, energies)
    E0 = expectation_value(gs, H)
    dEs = Es .- E0
    verbose && println("lehmann_report: $(length(excited)) excited state(s), $(length(idv)) source(s), E₀ = $E0")
    S = _lehmann_orthocheck(excited, orth_tol; verbose=verbose)
    cs = _lehmann_exact_overlaps(gs, op, idv, excited)
    _lehmann_overlap_check(cs, idv, dEs; overlap_abstol, overlap_warn_fraction, verbose=verbose)
    rec = nothing
    if all(e -> e isa Real, Es) && E0 isa Real
        rec = recommend_tmax(Es, E0; eta, verbose=verbose)
    end
    return (; cs=cs, dE=dEs, S=S, tmax=rec)
end
