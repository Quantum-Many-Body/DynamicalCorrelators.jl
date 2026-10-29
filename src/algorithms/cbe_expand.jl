# Controlled Bond Expansion (CBE) with factorized left/right complements.
#
# Jordan-MPO paths are grouped by the shared inner MPO index m into channels.
# The left half tensor L[m] sums the inner MPO index a BEFORE the complement
# projection (the projection is linear, so this is exact):
#   L[m] = P⟂( Σ_a T[a]·A1[(a,m)] + left·C1[m] ),   T[a] = GL[a]·left
# and the right half R[m] = ( Σ_b A2[(m,b)]·GR[b] + B2[m] )·P⟂ similarly, so
# the projected two-site update is x2 = Σ_m L[m]·R[m] — one GEMM per channel
# instead of one per path. No explicit nullspace basis is formed.
#
# With `rsvd = true` (and a `truncrank` truncation) the SVD runs on a sketched
# matrix, following FiniteMPS.jl's NaiveCBE: a random Ω of rank D_mid + D_add
# compresses the large side of x2 (the right side for Val(:right)), the
# sketched product x2t = Σ_m L·(R·Ω) yields an orthonormal basis Q, and only
# the compressed x2̂ = Σ_m (Q'·L)·R is factorized. When the sketch would not
# compress (small bonds) the full x2 is used instead, so `rsvd` is a pure
# performance switch. Val(:left) mirrors the scheme onto the left side.
#
# The graph shares T[a] = GL[a]·left and the per-channel halves L[m]/R[m] by
# their concrete keys through slots, like the AC2 tree; X2T/X2 nodes write the
# accumulators (the shared sketch x2t under a lock, resp. per-worker y).

"""
    CBEExpand(; trunc, alg_svd = SafeDivideAndConquer(), rsvd = true)

Controlled Bond Expansion (direct selection) as an MPSKit `changebond!`
bond-expansion algorithm, usable everywhere `OptimalExpand` is accepted — the
`alg_expand` keyword of [`dmrg1!`](@ref)/[`dmrg_mix!`](@ref), MPSKit's `DMRG`
and `TDVP` drivers, or a standalone `changebonds!` call.

The expansion directions are dominant singular vectors of the projected
two-site update. Jordan-MPO paths use factorized left/right complements and
MPSKit's `project_complement!`/`project_complement_right!`, avoiding explicit
nullspace bases and the full unprojected two-site vector. Other MPO types use
MPSKit's `OptimalExpand`.

`trunc` bounds how much is *added* to each bond, not the total bond dimension
that is kept. The projected block is normalized before the SVD, so only
`truncrank`/`truncspace` have a robust meaning here.

With `rsvd = true` and a `truncrank` truncation, the SVD runs on a sketched
matrix whose small dimension is `current bond dimension + added rank`
(FiniteMPS.jl's randomized CBE): a random sketch compresses the large side of
the projected block, an orthonormal basis of the sketch is built by QR/LQ, and
only the compressed block is factorized. When the sketch would not compress
(small bonds) the full block is used instead, so `rsvd` is a pure performance
switch; the expansion itself stays deterministic in exact mode.

# Fields
- `trunc::TruncationStrategy`: how many directions are *added* to each bond
- `alg_svd`: SVD algorithm (default: `SafeDivideAndConquer()`)
- `rsvd::Bool`: sketch the SVD target as described above (default: `true`)
"""
@kwdef struct CBEExpand{S} <: Algorithm
    alg_svd::S = SafeDivideAndConquer()
    trunc::TruncationStrategy
    rsvd::Bool = true
end

function _fc_t(GL, left)
    return @plansor T[-1 -2; -3 -4] := GL[-1 -2; 1] * left[1 -4; -3]
end

function _fc_la(T, A1)
    return @plansor L[-1 -2; -3 -4] := T[-1 1; -3 2] * A1[1 -2; 2 -4]
end

function _fc_lc(left, C1)
    return @plansor L[-1 -2; -3 -4] := left[-1 2; -3] * C1[-2; 2 -4]
end

function _fc_ra(right_tail, A2, GR)
    return @plansor R[-3 -4; -6 -5] :=
        right_tail[-3; 2 3] * A2[-4 -5; 3 6] * GR[2 6; -6]
end

function _fc_rb(right_tail, B2)
    return @plansor R[-3 -4; -5 -6] := right_tail[-3; -5 3] * B2[-4 -6; 3]
end

function _fc_x2(L, R)
    return @plansor x2[-1 -2; -5 -6] := L[-1 -2; 3 4] * R[3 4; -5 -6]
end

function _fc_x2!(y, L, R, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -5 -6] = L[-1 -2; 3 4] * R[3 4; -5 -6]
    else
        @plansor allocator = buffer y[-1 -2; -5 -6] += L[-1 -2; 3 4] * R[3 4; -5 -6]
    end
    return nothing
end

# rsvd kernels — Val(:right): sketch the (R1, R2) side of x2
function _fc_rΩ(R, Ω)
    return @plansor RΩ[-3 -4; -5] := R[-3 -4; 1 2] * Ω[1 2; -5]
end
function _fc_x2t(L, RΩ)
    return @plansor x2t[-1 -2; -3] := L[-1 -2; 3 4] * RΩ[3 4; -3]
end
function _fc_lq(Q, L)
    return @plansor LQ[-1; -3 -4] := conj(Q[1 2; -1]) * L[1 2; -3 -4]
end
function _fc_x2r(LQ, R)
    return @plansor y[-1; -5 -6] := LQ[-1; 3 4] * R[3 4; -5 -6]
end
function _fc_x2r!(y, LQ, R, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1; -5 -6] = LQ[-1; 3 4] * R[3 4; -5 -6]
    else
        @plansor allocator = buffer y[-1; -5 -6] += LQ[-1; 3 4] * R[3 4; -5 -6]
    end
    return nothing
end

# rsvd kernels — Val(:left): sketch the (D_left, p1) side of x2
function _fc_lΩ(Ω, L)
    return @plansor LΩ[-1; -3 -4] := Ω[-1; 1 2] * L[1 2; -3 -4]
end
function _fc_x2tl(LΩ, R)
    return @plansor x2t[-1; -5 -6] := LΩ[-1; 3 4] * R[3 4; -5 -6]
end
function _fc_rq(R, Q)
    return @plansor RQ[-3 -4; -5] := R[-3 -4; 1 2] * conj(Q[-5; 1 2])
end
function _fc_x2l(L, RQ)
    return @plansor y[-1 -2; -3] := L[-1 -2; 3 4] * RQ[3 4; -3]
end
function _fc_x2l!(y, L, RQ, buffer, first::Bool)
    if first
        @plansor allocator = buffer y[-1 -2; -3] = L[-1 -2; 3 4] * RQ[3 4; -3]
    else
        @plansor allocator = buffer y[-1 -2; -3] += L[-1 -2; 3 4] * RQ[3 4; -3]
    end
    return nothing
end

# Group Jordan-MPO paths into channels keyed by the shared inner MPO index m.
# The path families at m are Cartesian products of the two MPOs' block sets
# (AA: lA[m]×rA[m], AB: lA[m]×{B2[m]}, CA: {C1[m]}×rA[m], CB: {C1[m]}×{B2[m]}),
# so the halves are built directly from each MPO's blocks — one (a, A1) per
# W1.A block with I1[4] == m, one (b, A2, GR) per W2.A block with I2[1] == m —
# and NOT from path enumeration (which would duplicate a block that pairs with
# both W2.A and W2.B). lA[m] = [(a, A1b)], lC[m] = C1b, rA[m] = [(b, A2b, GRb)],
# rB[m] = B2b; a channel needs at least one left and one right half term.
# `rows` collects the left-bond indices a that need a T[a] = GL[a]·left slot.
function _fc_channels(W1, W2, GR2)
    lA = Dict{Int,Vector{Tuple{Int,Any}}}()
    lC = Dict{Int,Any}()
    rA = Dict{Int,Vector{Tuple{Int,Any,Any}}}()
    rB = Dict{Int,Any}()
    for (I1, A1b) in nonzero_pairs(W1.A)
        a, m = I1[1], I1[4]
        push!(get!(lA, m, Tuple{Int,Any}[]), (a, A1b))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C)
        lC[Ic[3]] = C1b
    end
    for (I2, A2b) in nonzero_pairs(W2.A)
        m, b = I2[1], I2[4]
        push!(get!(rA, m, Tuple{Int,Any,Any}[]), (b, A2b, GR2[b]))
    end
    for (Ib, B2b) in nonzero_pairs(W2.B)
        rB[Ib[1]] = B2b
    end
    channels = sort!(collect(intersect(
        union(keys(lA), keys(lC)), union(keys(rA), keys(rB)))))
    isempty(channels) && error("no CBE channels at this bond")
    rows = sort!(unique!([a for m in channels for (a, _) in get(lA, m, [])]))
    return channels, lA, lC, rA, rB, rows
end

# L = P⟂( Σ_a T[a]·A1[(a,m)] + left·C1[m] ) — the inner MPO index a is summed
# before the projection (project_complement! is linear, so this is exact).
function _fc_half_l(TA, C1b, left, left_c)
    L = nothing
    for (T, A1) in TA
        tmp = _fc_la(T, A1)
        L = L === nothing ? tmp : add!(L, tmp, One(), One())
    end
    if C1b !== nothing
        tmp = _fc_lc(left, C1b)
        L = L === nothing ? tmp : add!(L, tmp, One(), One())
    end
    return project_complement!(L, left_c)
end

function _fc_half_r(bterms, B2b, right_tail, right_c)
    R = nothing
    for (_, A2, GR) in bterms
        tmp = _fc_ra(right_tail, A2, GR)
        R = R === nothing ? tmp : add!(R, tmp, One(), One())
    end
    if B2b !== nothing
        tmp = _fc_rb(right_tail, B2b)
        R = R === nothing ? tmp : add!(R, tmp, One(), One())
    end
    return project_complement_right!(R, right_c)
end

function _fc_tnode(slot, GL, left)
    return (y, _, buf, first, state) -> _getslot!(state, slot) do
        _fc_t(GL, left)
    end
end

function _fc_lnode(slot, Tslots, aterms, C1b, left, left_c)
    return (y, _, buf, first, state) -> _getslot!(state, slot) do
        TA = [(state.slots[Tslots[ai]], A1) for (ai, A1) in aterms]
        _fc_half_l(TA, C1b, left, left_c)
    end
end

function _fc_rnode(slot, bterms, B2b, right_tail, right_c)
    return (y, _, buf, first, state) -> _getslot!(state, slot) do
        _fc_half_r(bterms, B2b, right_tail, right_c)
    end
end

function _fc_x2node(lslot, rslot)
    return (y, _, buf, first, state) ->
        _fc_x2!(y, state.slots[lslot], state.slots[rslot], buf, first)
end

# X2T node (rsvd): accumulate the sketched contraction L·(R·Ω) into the shared
# x2t; the lock serializes the (few, large) add!s across workers.
function _fc_x2tnode(Ω, x2t, lk, lslot, rslot)
    return (y, _, buf, first, state) -> begin
        tmp = _fc_x2t(state.slots[lslot], _fc_rΩ(state.slots[rslot], Ω))
        lock(lk)
        try
            add!(x2t, tmp, One(), One())
        finally
            unlock(lk)
        end
        return nothing
    end
end
function _fc_x2tlnode(Ω, x2t, lk, lslot, rslot)
    return (y, _, buf, first, state) -> begin
        tmp = _fc_x2tl(_fc_lΩ(Ω, state.slots[lslot]), state.slots[rslot])
        lock(lk)
        try
            add!(x2t, tmp, One(), One())
        finally
            unlock(lk)
        end
        return nothing
    end
end

# QR node (rsvd): runs after all X2T nodes, stores the sketch basis in Qref
# for the X2 nodes. left_orth!/right_orth! mutate x2t, which is dead by then.
_fc_qrnode(x2t, Qref, ::Val{:right}) =
    (y, _, buf, first, state) -> (Qref[] = left_orth!(x2t)[1]; nothing)
_fc_qrnode(x2t, Qref, ::Val{:left}) =
    (y, _, buf, first, state) -> (Qref[] = right_orth!(x2t)[2]; nothing)

function _fc_x2node_r(Qref, lslot, rslot)
    return (y, _, buf, first, state) -> _fc_x2r!(
        y, _fc_lq(Qref[], state.slots[lslot]), state.slots[rslot], buf, first)
end
function _fc_x2node_l(Qref, lslot, rslot)
    return (y, _, buf, first, state) -> _fc_x2l!(
        y, state.slots[lslot], _fc_rq(state.slots[rslot], Qref[]), buf, first)
end

function _fc_graph(
        GL2, channels, lA, lC, rA, rB, rows, left, right_tail, left_c, right_c, rsvd
    )
    builder = _GraphBuilder(:tree)
    Tslots = Dict{Int,Int}()
    Tnodes = Dict{Int,Int}()
    for a in rows
        slot = _add_slot!(builder)
        Tslots[a] = slot
        Tnodes[a] = _add_node!(builder, :T, 2, _fc_tnode(slot, GL2[a], left);
            writes_y = false, produces = [slot])
    end
    Lslots = Dict{Int,Int}()
    Rslots = Dict{Int,Int}()
    Lnodes = Dict{Int,Int}()
    Rnodes = Dict{Int,Int}()
    for m in channels
        aterms = get(lA, m, Tuple{Int,Any}[])
        C1b = get(lC, m, nothing)
        bterms = get(rA, m, Tuple{Int,Any,Any}[])
        B2b = get(rB, m, nothing)
        lslot = _add_slot!(builder)
        Lslots[m] = lslot
        Lnodes[m] = _add_node!(builder, :L, 3,
            _fc_lnode(lslot, Tslots, aterms, C1b, left, left_c);
            writes_y = false,
            deps = Int[Tnodes[a] for (a, _) in aterms],
            reads = Int[Tslots[a] for (a, _) in aterms],
            produces = [lslot])
        rslot = _add_slot!(builder)
        Rslots[m] = rslot
        Rnodes[m] = _add_node!(builder, :R, 3,
            _fc_rnode(rslot, bterms, B2b, right_tail, right_c);
            writes_y = false, produces = [rslot])
    end
    if rsvd === nothing
        for m in channels
            _add_node!(builder, :X2, 4, _fc_x2node(Lslots[m], Rslots[m]);
                deps = [Lnodes[m], Rnodes[m]], reads = [Lslots[m], Rslots[m]])
        end
    else
        (; Ω, x2t, lk, Qref) = rsvd
        x2tnodes = Vector{Int}(undef, length(channels))
        for (i, m) in enumerate(channels)
            apply = rsvd.dir isa Val{:right} ?
                _fc_x2tnode(Ω, x2t, lk, Lslots[m], Rslots[m]) :
                _fc_x2tlnode(Ω, x2t, lk, Lslots[m], Rslots[m])
            x2tnodes[i] = _add_node!(builder, :X2T, 4, apply;
                writes_y = false,
                deps = [Lnodes[m], Rnodes[m]], reads = [Lslots[m], Rslots[m]])
        end
        qr = _add_node!(builder, :QR, 1, _fc_qrnode(x2t, Qref, rsvd.dir);
            writes_y = false, deps = x2tnodes)
        for m in channels
            apply = rsvd.dir isa Val{:right} ?
                _fc_x2node_r(Qref, Lslots[m], Rslots[m]) :
                _fc_x2node_l(Qref, Lslots[m], Rslots[m])
            _add_node!(builder, :X2, 4, apply;
                deps = [qr, Lnodes[m], Rnodes[m]], reads = [Lslots[m], Rslots[m]])
        end
    end
    return _freeze_graph(builder)
end

# Truncated copy of fuse(V) with every sector dimension scaled by D/dim —
# FiniteMPS.jl's `_rsvd_trunc`: the sketch rank is distributed over sectors in
# proportion to their fused dimensions (rounded up), keeping the dual flag.
function _rsvd_space(V::ProductSpace{<:GradedSpace}, D::Int)
    Vf = fuse(V)
    ratio = D / dim(Vf)
    return Vect[sectortype(Vf)](
        ((isdual(Vf) ? dual(c) : c) => ceil(Int, dim(Vf, c) * ratio)
         for c in sectors(Vf)); dual = isdual(Vf))
end
function _rsvd_space(V::ProductSpace{<:ElementarySpace}, D::Int)
    Vf = fuse(V)
    return typeof(Vf)(D; dual = isdual(Vf))
end

# Real halves of the first channel: their spaces/eltypes fix the accumulators.
function _fc_example_halves(m, lA, lC, rA, rB, GL2, left, right_tail, left_c, right_c)
    TA = [(_fc_t(GL2[a], left), A1) for (a, A1) in get(lA, m, Tuple{Int,Any}[])]
    L_ex = _fc_half_l(TA, get(lC, m, nothing), left, left_c)
    R_ex = _fc_half_r(get(rA, m, Tuple{Int,Any,Any}[]), get(rB, m, nothing),
        right_tail, right_c)
    return L_ex, R_ex
end

# Random-sketch setup; `nothing` when the sketch would not compress, letting
# the caller fall back to the exact projected block.
function _fc_rsvd_setup(dir::Val{:right}, rsvd_rank, L_ex, R_ex)
    side = domain(R_ex)
    V = _rsvd_space(side, rsvd_rank)
    dim(V) ≥ dim(fuse(side)) && return nothing
    Ω = randn(promote_type(eltype(L_ex), eltype(R_ex)), side, V)
    xt_ex = _fc_x2t(L_ex, _fc_rΩ(R_ex, Ω))
    Q_ex = left_orth!(xt_ex)[1]
    y_ex = _fc_x2r(_fc_lq(Q_ex, L_ex), R_ex)
    return (; dir, Ω, x2t = vi_zerovector(xt_ex), lk = ReentrantLock(),
        Qref = Ref{Any}(), y = vi_zerovector(y_ex))
end
function _fc_rsvd_setup(dir::Val{:left}, rsvd_rank, L_ex, R_ex)
    side = codomain(L_ex)
    V = _rsvd_space(side, rsvd_rank)
    dim(V) ≥ dim(fuse(side)) && return nothing
    Ω = randn(promote_type(eltype(L_ex), eltype(R_ex)), V, side)
    xt_ex = _fc_x2tl(_fc_lΩ(Ω, L_ex), R_ex)
    Q_ex = right_orth!(xt_ex)[2]
    y_ex = _fc_x2l(L_ex, _fc_rq(R_ex, Q_ex))
    return (; dir, Ω, x2t = vi_zerovector(xt_ex), lk = ReentrantLock(),
        Qref = Ref{Any}(), y = vi_zerovector(y_ex))
end

function _fc_apply_all!(y, plan::_GraphPlan)
    state = _MatvecState(plan)
    workers = _WorkerResources()
    try
        return _apply_plan!(y, plan, state, workers,
            _ExecutionPolicy(cap = nothing, tree = true), y)
    finally
        _release_state!(state)
        _release_workers!(workers)
    end
end

function _fc_g2(site, ψ, H, envs, left, right, left_c, right_c, dir::Val, rsvd_rank)
    GL = leftenv(envs, site, ψ)
    GR = rightenv(envs, site + 1, ψ)
    W1, W2 = H[site], H[site + 1]
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    right_tail = _transpose_tail(right)
    channels, lA, lC, rA, rB, rows = _fc_channels(W1, W2, GR2)
    L_ex, R_ex = _fc_example_halves(
        first(channels), lA, lC, rA, rB, GL2, left, right_tail, left_c, right_c)
    rsvd = rsvd_rank === nothing ? nothing : _fc_rsvd_setup(dir, rsvd_rank, L_ex, R_ex)
    plan = _fc_graph(
        GL2, channels, lA, lC, rA, rB, rows, left, right_tail, left_c, right_c, rsvd)
    y = rsvd === nothing ? vi_zerovector(_fc_x2(L_ex, R_ex)) : rsvd.y
    return _fc_apply_all!(y, plan)
end

# Sketch rank D_mid + D_add (FiniteMPS.jl uses the total target rank, which is
# the same sum); only a plain `truncrank` gives a well-defined D_add.
_cbe_rsvd_rank(alg::CBEExpand, left) =
    alg.rsvd && alg.trunc isa TruncationByOrder ?
        dim(only(domain(left))) + alg.trunc.howmany : nothing

function _cbe_svd(t, alg::CBEExpand)
    if _SVD_THREADED[] && Threads.nthreads() > 1
        return threaded_svd_trunc(t, TruncatedAlgorithm(alg.alg_svd, alg.trunc))
    end
    return svd_trunc!(t; trunc = alg.trunc, alg = alg.alg_svd)
end

function changebond!(
        site::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, alg::CBEExpand, envs;
        normalize::Bool = true, allocator = nothing,
        timeroutput::TimerOutput = get_defaulttimer()
    )
    left = ψ.AC[site]
    right = ψ.AR[site + 1]
    if !(H isa MPOHamiltonian{<:JordanMPOTensor}) || numout(left) != 2
        return changebond!(site, Val(:right), ψ, H,
            OptimalExpand(; alg_svd = alg.alg_svd, trunc = alg.trunc), envs;
            normalize, allocator)
    end
    left_c = @timeit timeroutput "canonicalize" first(left_gauge(left))
    right_c = _transpose_tail(right)
    g2 = @timeit timeroutput "g2" _fc_g2(
        site, ψ, H, envs, left, right, left_c, right_c,
        Val(:right), _cbe_rsvd_rank(alg, left))
    nrm = norm(g2)
    nrm ≤ eps(real(scalartype(g2)))^(3 / 4) && return ψ
    _, _, Vᴴ = @timeit timeroutput "svd" _cbe_svd(scale!(g2, inv(nrm)), alg)

    @timeit timeroutput "reconstruct" begin
        nal_space = codomain(left) ← (only(domain(left)) ⊕ space(Vᴴ, 1))
        nal, nc, _ = left_gauge(absorb!(zerovector!(similar(left, nal_space)), left))
        nar = _transpose_front(catcodomain(_transpose_tail(right), Vᴴ))
        normalize && normalize!(nc)
        ψ.AC[site] = (nal, nc)
        ψ.AC[site + 1] = (nc, nar)
    end
    return ψ
end

function changebond!(
        site::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, alg::CBEExpand, envs;
        normalize::Bool = true, allocator = nothing,
        timeroutput::TimerOutput = get_defaulttimer()
    )
    left = ψ.AL[site - 1]
    right = ψ.AC[site]
    if !(H isa MPOHamiltonian{<:JordanMPOTensor}) || numout(left) != 2
        return changebond!(site, Val(:left), ψ, H,
            OptimalExpand(; alg_svd = alg.alg_svd, trunc = alg.trunc), envs;
            normalize, allocator)
    end
    left_c = left
    right_c = @timeit timeroutput "canonicalize" last(right_orth!(
        _transpose_tail(right; copy = true); trunc = notrunc()))
    g2 = @timeit timeroutput "g2" _fc_g2(
        site - 1, ψ, H, envs, left, right, left_c, right_c,
        Val(:left), _cbe_rsvd_rank(alg, left))
    nrm = norm(g2)
    nrm ≤ eps(real(scalartype(g2)))^(3 / 4) && return ψ
    U, _, _ = @timeit timeroutput "svd" _cbe_svd(scale!(g2, inv(nrm)), alg)

    @timeit timeroutput "reconstruct" begin
        right_tail = _transpose_tail(right)
        nc_space = (codomain(right_tail)[1] ⊕ _lastspace(U)') ← domain(right_tail)
        nc, Qr = lq_compact!(absorb!(zerovector!(similar(right_tail, nc_space)), right_tail))
        AL_exp = catdomain(left, U)
        normalize && normalize!(nc)
        ψ.AC[site] = (nc, _transpose_front(Qr))
        ψ.AC[site - 1] = (AL_exp, nc)
    end
    return ψ
end

changebonds(ψ::AbstractFiniteMPS, H, alg::CBEExpand, envs = environments(ψ, H, ψ)) =
    changebonds!(copy(ψ), H, alg, envs)

function changebonds!(
        ψ::AbstractFiniteMPS, H, alg::CBEExpand, envs = environments(ψ, H, ψ)
    )
    for i in 1:(length(ψ) - 1)
        changebond!(i, Val(:right), ψ, H, alg, envs)
    end
    return ψ, envs
end
