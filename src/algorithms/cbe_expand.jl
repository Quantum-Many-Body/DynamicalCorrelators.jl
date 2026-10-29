# Controlled Bond Expansion (CBE) with factorized left/right complements.
#
# Jordan-MPO paths are split at the MPO bond. Left half tensors are built from
# GL/left/W1, right half tensors from right_tail/W2/GR. MPSKit's complement
# projectors act on these halves directly, so no explicit nullspace basis is
# formed. The final x2 is assembled only from AA/CA/AB/CB channels; channels
# acting as the identity on either physical leg vanish under the projection.
#
# The graph shares T[a] = GL[a] * left and S[(a,m)] = T[a] * A1 only by their
# concrete keys, like the AC2 tree. CA/CB similarly reuse left * C1 only for
# the same C1 and m. Complete paths and final x2 contributions are never
# merged merely because they have the same block kind.

"""
    CBEExpand(; trunc, alg_svd = SafeDivideAndConquer())

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

# Fields
- `trunc::TruncationStrategy`: how many directions are *added* to each bond
- `alg_svd`: SVD algorithm (default: `SafeDivideAndConquer()`)
"""
@kwdef struct CBEExpand{S} <: Algorithm
    alg_svd::S = SafeDivideAndConquer()
    trunc::TruncationStrategy
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

function _fc_graph(GL2, W1, W2, GR2, left, right_tail, left_c, right_c)
    aa = NamedTuple[]
    ca = NamedTuple[]
    ab = NamedTuple[]
    cb = NamedTuple[]
    Tel = Union{}
    for (I1, A1b) in nonzero_pairs(W1.A), (I2, A2b) in nonzero_pairs(W2.A)
        I1[4] == I2[1] || continue
        a, m, b = I1[1], I1[4], I2[4]
        GLb, GRb = GL2[a], GR2[b]
        push!(aa, (; a, m, A1 = A1b, A2 = A2b, GR = GRb))
        Tel = promote_type(Tel, eltype(GLb), eltype(A1b), eltype(A2b), eltype(GRb))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ia, A2b) in nonzero_pairs(W2.A)
        Ic[3] == Ia[1] || continue
        m, b = Ic[3], Ia[4]
        GRb = GR2[b]
        push!(ca, (; m, C1 = C1b, A2 = A2b, GR = GRb))
        Tel = promote_type(Tel, eltype(C1b), eltype(A2b), eltype(GRb))
    end
    for (Ia, A1b) in nonzero_pairs(W1.A), (Ib, B2b) in nonzero_pairs(W2.B)
        Ia[4] == Ib[1] || continue
        a, m = Ia[1], Ia[4]
        GLb = GL2[a]
        push!(ab, (; a, m, A1 = A1b, B2 = B2b))
        Tel = promote_type(Tel, eltype(GLb), eltype(A1b), eltype(B2b))
    end
    for (Ic, C1b) in nonzero_pairs(W1.C), (Ib, B2b) in nonzero_pairs(W2.B)
        Ic[3] == Ib[1] || continue
        m = Ic[3]
        push!(cb, (; m, C1 = C1b, B2 = B2b))
        Tel = promote_type(Tel, eltype(C1b), eltype(B2b))
    end
    (isempty(aa) && isempty(ca) && isempty(ab) && isempty(cb)) &&
        error("no CBE channels at this bond")

    rows = sort!(unique!(vcat(Int[p.a for p in aa], Int[p.a for p in ab])))
    pairs = sort!(unique!(vcat(
        Tuple{Int,Int}[(p.a, p.m) for p in aa],
        Tuple{Int,Int}[(p.a, p.m) for p in ab]
    )))
    lc_keys = sort!(unique!(vcat(Int[p.m for p in ca], Int[p.m for p in cb])))
    ra_keys = sort!(unique!(vcat(
        Tuple{Int,Int}[(p.m, p.GR === nothing ? 0 : p.b) for p in aa],
        Tuple{Int,Int}[(p.m, p.GR === nothing ? 0 : p.b) for p in ca]
    )))
    rb_keys = sort!(unique!(vcat(Int[p.m for p in ab], Int[p.m for p in cb])))
    A1_of = Dict{Tuple{Int,Int},Any}()
    for p in aa
        A1_of[(p.a, p.m)] = p.A1
    end
    for p in ab
        A1_of[(p.a, p.m)] = p.A1
    end
    C1_of = Dict{Int,Any}()
    for p in ca
        C1_of[p.m] = p.C1
    end
    for p in cb
        C1_of[p.m] = p.C1
    end
    A2_of = Dict{Tuple{Int,Int},Any}()
    for p in aa
        A2_of[(p.m, p.b)] = (p.A2, p.GR)
    end
    for p in ca
        A2_of[(p.m, p.b)] = (p.A2, p.GR)
    end
    B2_of = Dict{Int,Any}()
    for p in ab
        B2_of[p.m] = p.B2
    end
    for p in cb
        B2_of[p.m] = p.B2
    end

    builder = _GraphBuilder(:tree)
    Tslots = Dict{Int,Int}()
    Sslots = Dict{Tuple{Int,Int},Int}()
    LCslots = Dict{Int,Int}()
    RAslots = Dict{Tuple{Int,Int},Int}()
    RBslots = Dict{Int,Int}()
    Tnodes = Dict{Int,Int}()
    Snodes = Dict{Tuple{Int,Int},Int}()
    LCnodes = Dict{Int,Int}()
    RAnodes = Dict{Tuple{Int,Int},Int}()
    RBnodes = Dict{Int,Int}()

    for a in rows
        slot = _add_slot!(builder)
        Tslots[a] = slot
        Tnodes[a] = _add_node!(builder, :T, 2,
            (y, _, buf, first, state) -> _getslot!(state, slot) do
                _fc_t(GL2[a], left)
            end;
            writes_y = false, produces = [slot])
    end
    for p in pairs
        a, m = p
        slot = _add_slot!(builder)
        Sslots[p] = slot
        Snodes[p] = _add_node!(builder, :S, 3,
            (y, _, buf, first, state) -> _getslot!(state, slot) do
                project_complement!(_fc_la(state.slots[Tslots[a]], A1_of[p]), left_c)
            end;
            writes_y = false, deps = [Tnodes[a]], reads = [Tslots[a]], produces = [slot])
    end
    for m in lc_keys
        slot = _add_slot!(builder)
        LCslots[m] = slot
        LCnodes[m] = _add_node!(builder, :LC, 2,
            (y, _, buf, first, state) -> _getslot!(state, slot) do
                project_complement!(_fc_lc(left, C1_of[m]), left_c)
            end;
            writes_y = false, produces = [slot])
    end
    for key in ra_keys
        slot = _add_slot!(builder)
        RAslots[key] = slot
        A2, GR = A2_of[key]
        RAnodes[key] = _add_node!(builder, :RA, 3,
            (y, _, buf, first, state) -> _getslot!(state, slot) do
                project_complement_right!(_fc_ra(right_tail, A2, GR), right_c)
            end;
            writes_y = false, produces = [slot])
    end
    for m in rb_keys
        slot = _add_slot!(builder)
        RBslots[m] = slot
        RBnodes[m] = _add_node!(builder, :RB, 2,
            (y, _, buf, first, state) -> _getslot!(state, slot) do
                project_complement_right!(_fc_rb(right_tail, B2_of[m]), right_c)
            end;
            writes_y = false, produces = [slot])
    end

    for p in aa
        lkey, rkey = (p.a, p.m), (p.m, p.b)
        _add_node!(builder, :AA, 4,
            (y, _, buf, first, state) -> _fc_x2!(
                y, state.slots[Sslots[lkey]], state.slots[RAslots[rkey]], buf, first
            );
            deps = [Snodes[lkey], RAnodes[rkey]], reads = [Sslots[lkey], RAslots[rkey]])
    end
    for p in ca
        rkey = (p.m, p.b)
        _add_node!(builder, :CA, 4,
            (y, _, buf, first, state) -> _fc_x2!(
                y, state.slots[LCslots[p.m]], state.slots[RAslots[rkey]], buf, first
            );
            deps = [LCnodes[p.m], RAnodes[rkey]], reads = [LCslots[p.m], RAslots[rkey]])
    end
    for p in ab
        lkey = (p.a, p.m)
        _add_node!(builder, :AB, 4,
            (y, _, buf, first, state) -> _fc_x2!(
                y, state.slots[Sslots[lkey]], state.slots[RBslots[p.m]], buf, first
            );
            deps = [Snodes[lkey], RBnodes[p.m]], reads = [Sslots[lkey], RBslots[p.m]])
    end
    for p in cb
        _add_node!(builder, :CB, 4,
            (y, _, buf, first, state) -> _fc_x2!(
                y, state.slots[LCslots[p.m]], state.slots[RBslots[p.m]], buf, first
            );
            deps = [LCnodes[p.m], RBnodes[p.m]], reads = [LCslots[p.m], RBslots[p.m]])
    end
    return _freeze_graph(builder), aa, ca, ab, cb
end

function _fc_example(aa, ca, ab, cb, GL2, left, right_tail)
    if !isempty(aa)
        p = first(aa)
        return _fc_x2(_fc_la(_fc_t(GL2[p.a], left), p.A1), _fc_ra(right_tail, p.A2, p.GR))
    elseif !isempty(ca)
        p = first(ca)
        return _fc_x2(_fc_lc(left, p.C1), _fc_ra(right_tail, p.A2, p.GR))
    elseif !isempty(ab)
        p = first(ab)
        return _fc_x2(_fc_la(_fc_t(GL2[p.a], left), p.A1), _fc_rb(right_tail, p.B2))
    else
        p = first(cb)
        return _fc_x2(_fc_lc(left, p.C1), _fc_rb(right_tail, p.B2))
    end
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

function _fc_g2(site, ψ, H, envs, left, right, left_c, right_c)
    GL = leftenv(envs, site, ψ)
    GR = rightenv(envs, site + 1, ψ)
    W1, W2 = H[site], H[site + 1]
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    right_tail = _transpose_tail(right)
    plan, aa, ca, ab, cb = _fc_graph(GL2, W1, W2, GR2, left, right_tail, left_c, right_c)
    example = _fc_example(aa, ca, ab, cb, GL2, left, right_tail)
    y = vi_zerovector(example)
    return _fc_apply_all!(y, plan)
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
    g2 = @timeit timeroutput "g2" _fc_g2(site, ψ, H, envs, left, right, left_c, right_c)
    nrm = norm(g2)
    nrm ≤ eps(real(scalartype(g2)))^(3 / 4) && return ψ
    _, _, Vᴴ = @timeit timeroutput "svd" svd_trunc!(
        scale!(g2, inv(nrm)); trunc = alg.trunc, alg = alg.alg_svd)

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
    g2 = @timeit timeroutput "g2" _fc_g2(site - 1, ψ, H, envs, left, right, left_c, right_c)
    nrm = norm(g2)
    nrm ≤ eps(real(scalartype(g2)))^(3 / 4) && return ψ
    U, _, _ = @timeit timeroutput "svd" svd_trunc!(
        scale!(g2, inv(nrm)); trunc = alg.trunc, alg = alg.alg_svd)

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
