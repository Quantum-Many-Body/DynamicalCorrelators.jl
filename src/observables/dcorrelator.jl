"""
    evolve_mps(H::MPOHamiltonian, ts::AbstractVector,
               rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); kwargs...)

Evolve `rho_mps` through the time grid `ts` under a time-independent
Hamiltonian `H`.

The first `n` recorded steps use `tdvp2` and later steps use `tdvp1`. Saved
states are written to `filename` for every index in `save_id` with keys
`"t=\$(ts[i])"`. For finite-temperature `dcorrelator`, save every requested
real-time slice, for example `save_id = eachindex(times)`.
"""
function evolve_mps(H::MPOHamiltonian, ts::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); 
                    filename::String="default_expiHt_ψ.jld2", 
                    save_id::AbstractArray=[length(ts),], 
                    verbose::Bool=true, 
                    n::Integer=3, 
                    tdvp1 = myTDVP(),
                    tdvp2 = myTDVP2(; trunc=truncerror(; rtol=1e-3))
                    )
    timer = TimerOutput()
    start_time, record_start = now(), now()
    verbose && _progress_start(1, length(ts), "t = $(ts[1])")
    flush(stdout)
    envs = environments(rho_mps, H)
    jldopen(filename, "w") do f
        f["ts"] = ts
        if 1 in save_id
            f["t=$(ts[1])"] = rho_mps
        end
    end
    for i in 2:length(ts)
        alg = i > n ? tdvp1 : tdvp2
        rho_mps, envs = _timed_timestep(timer, rho_mps, H, 0, ts[i]-ts[i-1], alg, envs)
        current_time = now()
        verbose && _progress_step(i, length(ts), "t = $(ts[i])", current_time - start_time)
        flush(stdout)
        jldopen(filename, "a") do f
            if i in save_id
                f["t=$(ts[i])"] = rho_mps
            end
        end
        start_time = current_time
    end
    verbose && _progress_end(record_start, timer)
    return rho_mps
end

"""
    evolve_mps(H::Function, ts::AbstractVector, mus::AbstractVector,
               rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H(mus[1]))); kwargs...)

Evolve `rho_mps` through `ts` under a time-dependent Hamiltonian `H(mu)`.

At step `i`, the Hamiltonian is rebuilt as `H(mus[i])`. Saved states and the
corresponding parameter values are written to `filename` with keys
`"t=\$(ts[i])"` and `"mu_t=\$(ts[i])"`. For finite-temperature `dcorrelator`,
save every requested real-time slice, for example `save_id = eachindex(times)`.
"""
function evolve_mps(H::Function, ts::AbstractVector, mus::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H(mus[1]))); 
                    filename::String="default_expiHt_ψ.jld2", 
                    save_id::AbstractArray=[length(ts),], 
                    verbose::Bool=true, 
                    n::Integer=3, 
                    tdvp1 = myTDVP(),
                    tdvp2 = myTDVP2(; trunc=truncerror(; rtol=1e-3))
                    )
    timer = TimerOutput()
    start_time, record_start = now(), now()
    verbose && _progress_start(1, length(ts), "t = $(ts[1])")
    flush(stdout)
    H0 = @timeit timer "build Hamiltonian" H(mus[1])
    envs = environments(rho_mps, H0)
    jldopen(filename, "w") do f
        f["ts"] = ts
        if 1 in save_id
            f["t=$(ts[1])"] = rho_mps
            f["mu_t=$(ts[1])"] = mus[1]
        end
    end
    for i in 2:length(ts)
        alg = i > n ? tdvp1 : tdvp2
        Hi = @timeit timer "build Hamiltonian" H(mus[i])
        rho_mps, envs = _timed_timestep(timer, rho_mps, Hi, 0, ts[i]-ts[i-1], alg, envs)
        current_time = now()
        verbose && _progress_step(i, length(ts), "t = $(ts[i])", current_time - start_time)
        flush(stdout)
        jldopen(filename, "a") do f
            if i in save_id
                f["t=$(ts[i])"] = rho_mps
                f["mu_t=$(ts[i])"] = mus[i]
            end
        end
        start_time = current_time
    end
    verbose && _progress_end(record_start, timer)
    return rho_mps
end

function _short_duration(period::Dates.CompoundPeriod)
    parts = String[]
    for p in period.periods
        value = Dates.value(p)
        value == 0 && continue
        if p isa Dates.Day
            push!(parts, "$(value)d")
        elseif p isa Dates.Hour
            push!(parts, "$(value)h")
        elseif p isa Dates.Minute
            push!(parts, "$(value)m")
        elseif p isa Dates.Second
            push!(parts, "$(value)s")
        elseif p isa Dates.Millisecond
            push!(parts, "$(value)ms")
        end
    end
    return isempty(parts) ? "0ms" : join(parts, ", ")
end

_short_duration(period::Dates.Period) = _short_duration(Dates.canonicalize(period))

_progress_prefix(i::Integer, n::Integer) = "[$(lpad(i, ndigits(n)))/$n]"

function _progress_start(i::Integer, n::Integer, message)
    println(_progress_prefix(i, n), " ", message, " | started: ", Dates.format(now(), "d.u yyyy HH:MM"))
end

function _progress_step(i::Integer, n::Integer, message, duration)
    println(_progress_prefix(i, n), " ", message, " | duration: ", _short_duration(duration))
end

function _progress_end(start_time, timer::TimerOutput)
    println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", _short_duration(now() - start_time))
    println(timer)
end

function _timed_timestep(timer::TimerOutput, ψ, H, t, dt, alg, envs)
    @timeit timer "time loop / timestep" begin
        if alg isa TDVP1_CBE
            return timestep(ψ, H, t, dt, alg, envs; timer)
        else
            return timestep(ψ, H, t, dt, alg, envs)
        end
    end
end

function _dcorrelator_record_window(times, record_indices)
    record_indices = collect(record_indices)
    isempty(record_indices) && throw(ArgumentError("record_indices cannot be empty"))
    record_indices == collect(record_indices[1]:record_indices[end]) ||
        throw(ArgumentError("record_indices must be continuous"))
    1 <= record_indices[1] <= record_indices[end] <= length(times) ||
        throw(ArgumentError("record_indices must be inside 1:$(length(times))"))
    return record_indices, record_indices[1], record_indices[end]
end

function _dcorrelator_site(id::Integer, L::Integer; doubled::Bool=true)
    upper = doubled ? 2L : L
    1 <= id <= upper || throw(ArgumentError("id must be inside 1:$(upper)"))
    return id <= L ? id : id - L
end

function _dcorrelator_load_complete!(gf, filename::String, nrecords::Integer; verbose::Bool=false)
    isfile(filename) || return false
    gfb = load(filename)
    iscomplete = all("pro_$(r)" in keys(gfb) for r in 1:nrecords)
    if iscomplete
        for r in 1:nrecords
            gf[:, r] = gfb["pro_$(r)"]
        end
        verbose && println("$(basename(filename)) has existed!")
        verbose && flush(stdout)
        return true
    end
    @warn "$(filename) is incomplete; recomputing it"
    return false
end

_dcorrelator_rho_key(t) = "t=$(t)"

function _dcorrelator_load_rho(rho_path::AbstractString, t)
    isfile(rho_path) || throw(ArgumentError("rho_path does not exist: $(rho_path)"))
    return load(rho_path, _dcorrelator_rho_key(t))
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian,
                op::AbstractTensorMap, id::Integer; kwargs...)

Compute a zero-temperature dynamical correlation from one source site.

`id` selects the source channel. Values `1:length(H)` use the forward
operator channel, while `length(H)+1:2length(H)` use the conjugated channel.
Only `record_indices` are returned, and the evolution stops at the last
requested record. This single-source method uses ordinary arrays and does not
start a distributed `SharedArray` calculation.
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, id::Integer;
                    verbose=true,
                    gf_path::String="./",
                    times::AbstractRange=0:0.05:5.0,
                    record_indices::AbstractArray=1:length(times),
                    n::Integer=3,
                    approxalg = myDMRG2(),
                    tdvp1 = myTDVP(),
                    tdvp2 = myTDVP2(; trunc=truncerror(; rtol=1e-3))
                    )
    !isdir(gf_path) && mkpath(gf_path)
    L = length(H)
    idx = _dcorrelator_site(id, L)
    record_indices, record_first, record_last = _dcorrelator_record_window(times, record_indices)
    gsenergy = expectation_value(gs, H)
    gf = zeros(ComplexF64, L, length(record_indices))
    filename = joinpath(gf_path, "gf_start=$(times[record_first])_end=$(times[record_last])_id=$(id).jld2")

    _dcorrelator_load_complete!(gf, filename, length(record_indices); verbose=verbose) && return gf

    timer = TimerOutput()
    ket = @timeit timer "setup / chargedMPS" chargedMPS(op, gs, idx, approxalg)
    start_time, wall_start = now(), now()
    if record_first == 1
        phase = id <= L ? exp(im*gsenergy*times[1]) : exp(-im*gsenergy*times[1])
        sd = @timeit timer "sweep_dot" sweep_dot(gs, op, ket)
        gf[:, 1] = id <= L ? -im * phase .* sd : -im * phase .* conj.(sd)
    end
    jldopen(filename, "w") do f
        f["times"] = times
        f["record_indices"] = record_indices
        f["id"] = id
        if record_first == 1
            f["pro_1"] = gf[:, 1]
        end
    end
    verbose && _progress_start(1, length(times), "time evolves 0.0 of ket$(id)")
    verbose && flush(stdout)

    envs = environments(ket, H)
    for k in 2:record_last
        alg = k > n ? tdvp1 : tdvp2
        ket, envs = _timed_timestep(timer, ket, H, 0, times[k] - times[k - 1], alg, envs)
        current_time = now()
        verbose && _progress_step(k, length(times), "time evolves $(times[k]) of ket$(id)", current_time - start_time)
        verbose && flush(stdout)
        if k >= record_first
            r = k - record_first + 1
            phase = id <= L ? exp(im*gsenergy*times[k]) : exp(-im*gsenergy*times[k])
            sd = @timeit timer "sweep_dot" sweep_dot(gs, op, ket)
            gf[:, r] = id <= L ? -im * phase .* sd : -im * phase .* conj.(sd)
            jldopen(filename, "a") do f
                f["pro_$(r)"] = gf[:, r]
            end
        end
        start_time = current_time
    end

    ket = nothing
    envs = nothing
    GC.gc()
    verbose && _progress_end(wall_start, timer)
    return gf
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian,
                op::AbstractTensorMap, indices::AbstractArray; kwargs...)

Compute zero-temperature dynamical correlations for many source channels.

The source channels in `indices` are evaluated in parallel with Distributed and
stored in a `SharedArray`. The result has size
`(length(H), length(indices), length(record_indices))`. Completed per-source
JLD2 files are reused; incomplete files are recomputed.
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true,
                    gf_path::String="./",
                    times::AbstractRange=0:0.05:5.0,
                    record_indices::AbstractArray=1:length(times),
                    n::Integer=3,
                    approxalg = myDMRG2(),
                    tdvp1 = myTDVP(),
                    tdvp2 = myTDVP2(; trunc=truncerror(; rtol=1e-3))
                    )
    !isdir(gf_path) && mkdir(gf_path)
    record_indices = collect(record_indices)
    isempty(record_indices) && throw(ArgumentError("record_indices cannot be empty"))
    record_indices == collect(record_indices[1]:record_indices[end]) ||
        throw(ArgumentError("record_indices must be continuous"))
    1 <= record_indices[1] <= record_indices[end] <= length(times) ||
        throw(ArgumentError("record_indices must be inside 1:$(length(times))"))
    record_first, record_last = record_indices[1], record_indices[end]
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(H), length(indices), length(record_indices))
    @sync @distributed for d in eachindex(indices)
        id = indices[d]
        idx = id <= length(H) ? id : (id - length(H))
        filename = joinpath(gf_path, "gf_start=$(times[record_first])_end=$(times[record_last])_id=$(id).jld2")
        if isfile(filename)
            gfb = load(filename)
            iscomplete = all("pro_$(r)" in keys(gfb) for r in 1:length(record_indices))
            if iscomplete
                for r in 1:length(record_indices)
                    gf[:,d,r] = gfb["pro_$(r)"]
                end
                verbose && println("gf_start=$(times[record_first])_end=$(times[record_last])_id=$(id).jld2 has existed!")
                flush(stdout)
                continue
            end
            @warn "$(filename) is incomplete; recomputing it"
        end
        timer = TimerOutput()
        ket = @timeit timer "setup / chargedMPS" chargedMPS(op, gs, idx, approxalg)
        start_time, wall_start = now(), now()
        if record_first == 1
            phase = id <= length(H) ? exp(im*gsenergy*times[1]) : exp(-im*gsenergy*times[1])
            sd = @timeit timer "sweep_dot" sweep_dot(gs, op, ket)
            gf[:,d,1] = id <= length(H) ? -im * phase .* sd : -im * phase .* conj.(sd)
        end
        jldopen(filename, "w") do f
            if record_first == 1
                f["pro_1"] = gf[:,d,1]
            end
        end
        verbose && _progress_start(1, length(times), "time evolves 0.0 of ket$(id)")
        flush(stdout)
        envs = environments(ket, H)
        for k in 2:record_last
            alg = k > n ? tdvp1 : tdvp2
            ket, envs = _timed_timestep(timer, ket, H, 0, times[k]-times[k-1], alg, envs)
            if k >= record_first
                r = k - record_first + 1
                phase = id <= length(H) ? exp(im*gsenergy*times[k]) : exp(-im*gsenergy*times[k])
                sd = @timeit timer "sweep_dot" sweep_dot(gs, op, ket)
                gf[:,d,r] = id <= length(H) ? -im * phase .* sd : -im * phase .* conj.(sd)
                current_time = now()
                verbose && _progress_step(k, length(times), "time evolves $(times[k]) of ket$(id)", current_time - start_time)
                flush(stdout)
                jldopen(filename, "a") do f
                    f["pro_$(r)"] = gf[:,d,r]
                end
            else
                current_time = now()
                verbose && _progress_step(k, length(times), "time evolves $(times[k]) of ket$(id)", current_time - start_time)
                flush(stdout)
            end
            start_time = current_time
        end
        ket = nothing
        envs = nothing
        GC.gc()
        verbose && _progress_end(wall_start, timer)
    end
    gfs = zeros(ComplexF64, length(H), length(indices), length(record_indices))
    gfs .= gf
    return gfs
end

"""
    dcorrelator(rho_path::AbstractString, H::MPOHamiltonian,
                op::AbstractTensorMap, id::Integer; kwargs...)

Compute a finite-temperature dynamical correlation from one source channel.

`rho_path` must point to a JLD2 trajectory produced by `evolve_mps`, with
thermal states stored under keys `"t=\$(times[k])"`. The source ket is evolved
in memory, while each `rho(t)` is loaded from disk only for the current
correlator slice. The result has size `(length(H), length(times))` and is
multiplied by `-im` before returning.
"""
function dcorrelator(rho_path::AbstractString, H::MPOHamiltonian, op::AbstractTensorMap, id::Integer;
                    verbose=true,
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    n::Integer=3, 
                    tdvp1 = myTDVP(),
                    tdvp2 = myTDVP2(; trunc=truncerror(; rtol=1e-3)),
                    )
    !isdir(gf_path) && mkpath(gf_path)
    L, nt = length(H), length(times)
    idx = _dcorrelator_site(id, L)
    gf = zeros(ComplexF64, L, nt)
    filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2")

    _dcorrelator_load_complete!(gf, filename, nt; verbose=verbose) && return -im * gf

    timer = TimerOutput()
    rho = @timeit timer "load rho" _dcorrelator_load_rho(rho_path, times[1])
    Z = dot(rho, rho)
    ket = @timeit timer "setup / chargedMPS" chargedMPS(op, rho, idx)
    ket_env = environments(ket, H)
    wall_start = now()

    jldopen(filename, "w") do f
        f["times"] = times
        f["id"] = id
        f["beta"] = beta
    end

    for k in 1:nt
        step_start = now()
        rho_t = k == 1 ? rho : (@timeit timer "load rho" _dcorrelator_load_rho(rho_path, times[k]))
        sd = @timeit timer "sweep_dot" sweep_dot(rho_t, op, ket)
        gf[:, k] = id <= L ? sd ./ Z : conj.(sd) ./ Z

        jldopen(filename, "a") do f
            f["pro_$(k)"] = gf[:, k]
        end

        current_time = now()
        verbose && _progress_step(k, nt, "finite-T correlation t=$(times[k]) of ket$(id)", current_time - step_start)
        verbose && flush(stdout)

        if k < nt
            alg = (k + 1) > n ? tdvp1 : tdvp2
            dt = times[k + 1] - times[k]
            ket, ket_env = _timed_timestep(timer, ket, H, 0, dt, alg, ket_env)
        end
    end

    rho = nothing
    ket = nothing
    ket_env = nothing
    GC.gc()
    verbose && _progress_end(wall_start, timer)
    return -im * gf
end

"""
    dcorrelator(rho_path::AbstractString, H::MPOHamiltonian,
                op::AbstractTensorMap, indices::AbstractArray; kwargs...)

Compute finite-temperature dynamical correlations for several source channels.

The source channels in `indices` are evaluated independently, following the
zero-temperature multi-source layout. Each worker keeps only one charged source
ket, its environment, and the current loaded `rho(t)` in memory. Completed
per-source JLD2 files are loaded and skipped; incomplete files are recomputed.
"""
function dcorrelator(rho_path::AbstractString, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    n::Integer=3, 
                    tdvp1 = myTDVP(),
                    tdvp2 = myTDVP2(; trunc=truncerror(; rtol=1e-3)),
                    )
    !isdir(gf_path) && mkpath(gf_path)
    L, nt = length(H), length(times)
    ids = collect(indices)
    gf = SharedArray{ComplexF64, 3}(L, length(ids), nt)

    @sync @distributed for d in eachindex(ids)
        id = ids[d]
        idx = _dcorrelator_site(id, L)
        filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2")

        if isfile(filename)
            gfb = load(filename)
            iscomplete = all("pro_$(k)" in keys(gfb) for k in 1:nt)
            if iscomplete
                for k in 1:nt
                    gf[:, d, k] = gfb["pro_$(k)"]
                end
                verbose && println("gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2 has existed!")
                verbose && flush(stdout)
                continue
            end
            @warn "$(filename) is incomplete; recomputing it"
        end

        timer = TimerOutput()
        rho = @timeit timer "load rho" _dcorrelator_load_rho(rho_path, times[1])
        Z = dot(rho, rho)
        ket = @timeit timer "setup / chargedMPS" chargedMPS(op, rho, idx)
        ket_env = environments(ket, H)
        wall_start = now()

        jldopen(filename, "w") do f
            f["times"] = times
            f["id"] = id
            f["beta"] = beta
        end

        for k in 1:nt
            step_start = now()
            rho_t = k == 1 ? rho : (@timeit timer "load rho" _dcorrelator_load_rho(rho_path, times[k]))
            sd = @timeit timer "sweep_dot" sweep_dot(rho_t, op, ket)
            gf[:, d, k] = id <= L ? sd ./ Z : conj.(sd) ./ Z

            jldopen(filename, "a") do f
                f["pro_$(k)"] = gf[:, d, k]
            end

            current_time = now()
            verbose && _progress_step(k, nt, "finite-T correlation t=$(times[k]) of ket$(id)", current_time - step_start)
            verbose && flush(stdout)

            if k < nt
                alg = (k + 1) > n ? tdvp1 : tdvp2
                dt = times[k + 1] - times[k]
                ket, ket_env = _timed_timestep(timer, ket, H, 0, dt, alg, ket_env)
            end
        end

        rho = nothing
        ket = nothing
        ket_env = nothing
        GC.gc()
        verbose && _progress_end(wall_start, timer)
    end

    gfs = zeros(ComplexF64, L, length(ids), nt)
    gfs .= gf
    return -im * gfs
end

"""
    sweep_dot(gs::FiniteNormalMPS, op::AbstractTensorMap, ket::FiniteNormalMPS) -> Vector

Compute `⟨gs|chargedMPO(op,i)†|ket⟩` for all sites `i=1..L` in a single sweep.

# Arguments
- `gs`: the ground state MPS.
- `op`: the local operator. Allowed leg structures:
  - `(1,2)`: `op[-1; -2 -3]` with MPO virtual leg on the right (side=:L convention).
  - `(1,1)`: `op[-1; -2]` diagonal operator without virtual leg.
  Operators with `(2,1)` legs (virtual leg on the left) are not supported.
- `ket`: the time-evolved charged MPS, e.g. `ket = chargedMPS(op, gs, j, approxalg)`.
"""
function sweep_dot(gs::FiniteNormalMPS, op::AbstractTensorMap, ket::FiniteNormalMPS)
    L = length(gs)
    S = promote_type(scalartype(gs), scalartype(ket))
    ncodom = length(codomain(op))
    ndom = length(domain(op))

    if ncodom == 2 && ndom == 1
        throw(ArgumentError("operator with virtual leg on the codomain is not supported; use side=:L convention"))
    end

    if ncodom == 1 && ndom == 2
        # Non-trivial virtual leg: chargedMPO = [I..I | op | Z..Z]
        # gr is MPSTensor (2 codomain, 1 domain), gl is MPSBondTensor (1,1)
        Z = fZ(op)
        gr = Vector{MPSTensor}(undef, L + 1)
        gr[L + 1] = isomorphism(S, right_virtualspace(gs, L) ⊗ domain(Z, 2) ← right_virtualspace(ket, L))
        for j in L:-1:1
            gr[j] = transfer_right(gr[j + 1], Z, gs.AR[j], ket.AR[j])
        end

        gl = Vector{MPSBondTensor}(undef, L + 1)
        gl[1] = isomorphism(S, left_virtualspace(ket, 1) ← left_virtualspace(gs, 1))
        for j in 1:L
            gl[j + 1] = transfer_left(gl[j], gs.AL[j], ket.AL[j])
        end

        G = zeros(S, L)
        for i in 1:L
            @plansor Gi = gl[i][1; 2] * gs.AC[i][2 3; 4] * gr[i+1][4 6; 7] * op[5; 3 6] * conj(ket.AC[i][1 5; 7])
            G[i] = conj(Gi)
        end

    elseif ncodom == 1 && ndom == 1
        # Diagonal operator: chargedMPO = [I..I | op | I..I], no Z string
        # Both gl and gr are MPSBondTensor (1,1)
        gr = Vector{MPSBondTensor}(undef, L + 1)
        gr[L + 1] = isomorphism(S, right_virtualspace(gs, L) ← right_virtualspace(ket, L))
        for j in L:-1:1
            gr[j] = transfer_right(gr[j + 1], gs.AR[j], ket.AR[j])
        end

        gl = Vector{MPSBondTensor}(undef, L + 1)
        gl[1] = isomorphism(S, left_virtualspace(ket, 1) ← left_virtualspace(gs, 1))
        for j in 1:L
            gl[j + 1] = transfer_left(gl[j], gs.AL[j], ket.AL[j])
        end

        G = zeros(S, L)
        for i in 1:L
            @plansor Gi = gl[i][1; 2] * gs.AC[i][2 3; 4] * gr[i+1][4; 7] * op[5; 3] * conj(ket.AC[i][1 5; 7])
            G[i] = conj(Gi)
        end

    else
        throw(ArgumentError("unsupported operator structure ($(ncodom),$(ndom))"))
    end

    return G
end

"""
    sweep_dot(rho::FiniteSuperMPS, op::AbstractTensorMap, ket::FiniteSuperMPS) -> Vector

Compute `dot(chargedMPS(op, rho, i), ket)` for every site of a finite-temperature
super MPS without explicitly constructing each charged state.

Supported operator structures are `(1,2)` using the side `:L` convention and
charge-neutral `(1,1)` operators. Operators with `(2,1)` legs are intentionally
rejected to match the zero-temperature sweep convention.
"""
function sweep_dot(rho::FiniteSuperMPS, op::AbstractTensorMap, ket::FiniteSuperMPS)
    L = length(rho)
    length(ket) == L || throw(ArgumentError("rho and ket must have the same length"))

    S = promote_type(scalartype(rho), scalartype(op), scalartype(ket))
    ncodom = length(codomain(op))
    ndom = length(domain(op))

    if ncodom == 2 && ndom == 1
        throw(ArgumentError("operator with virtual leg on the codomain is not supported; use side=:L convention"))
    end

    if ncodom == 1 && ndom == 2
        gr_end = isomorphism(S, right_virtualspace(rho, L) ⊗ domain(op, 2) ← right_virtualspace(ket, L))
        gr = Vector{typeof(gr_end)}(undef, L + 1)
        gr[L + 1] = gr_end

        t1 = TensorMap(BraidingTensor(domain(op, 1), domain(op, 2)))
        t2 = TensorMap(BraidingTensor(dual(domain(op, 1)), domain(op, 2)))
        for j in L:-1:1
            gr[j] = transfer_right(gr[j + 1], t1, t2, rho.AR[j], ket.AR[j])
        end

        gl_start = isomorphism(S, left_virtualspace(ket, 1) ← left_virtualspace(rho, 1))
        gl = Vector{typeof(gl_start)}(undef, L + 1)
        gl[1] = gl_start
        for j in 1:L
            gl[j + 1] = transfer_left(gl[j], rho.AL[j], ket.AL[j])
        end

        G = zeros(S, L)
        for i in 1:L
            @plansor Gi = gl[i][10; 1] *
                rho.AC[i][1 2 3; 4] *
                op[5; 2 6] *
                τ[6 7; 3 8] *
                gr[i + 1][4 8; 9] *
                conj(ket.AC[i][10 5 7; 9])
            G[i] = conj(Gi)
        end

    elseif ncodom == 1 && ndom == 1
        gr_end = isomorphism(S, right_virtualspace(rho, L) ← right_virtualspace(ket, L))
        gr = Vector{typeof(gr_end)}(undef, L + 1)
        gr[L + 1] = gr_end
        for j in L:-1:1
            gr[j] = transfer_right(gr[j + 1], rho.AR[j], ket.AR[j])
        end

        gl_start = isomorphism(S, left_virtualspace(ket, 1) ← left_virtualspace(rho, 1))
        gl = Vector{typeof(gl_start)}(undef, L + 1)
        gl[1] = gl_start
        for j in 1:L
            gl[j + 1] = transfer_left(gl[j], rho.AL[j], ket.AL[j])
        end

        G = zeros(S, L)
        for i in 1:L
            @plansor Gi = gl[i][6; 1] *
                rho.AC[i][1 2 3; 4] *
                op[5; 2] *
                conj(ket.AC[i][6 5 3; 7]) * gr[i + 1][4; 7]
            G[i] = conj(Gi)
        end

    else
        throw(ArgumentError("unsupported operator structure ($(ncodom),$(ndom))"))
    end

    return G
end
