"""
    MomentumMPS

Momentum-window MPS (q-MPS) state of Van Damme and Vanderstraeten.

The object is deliberately a thin wrapper around a collection of MPSKit
`WindowMPS` objects.  `windowmps` contains one factorized block for each
inequivalent site in the background unit cell.  A one-site unit cell therefore
has a one-element collection.

```text
    V_l -- X_1 -- X_2 -- ... -- X_N,
```

so every entry is an ordinary three-leg MPS tensor with one physical leg.  The
extended block tensor called `B` in Appendix A of the paper is obtained by
contracting this chain; it is not a single-site tensor and `X_1` is not a bond
tensor.

The first tensor is a left-null isometry of the background tensor `A_l`:

```math
    A_l^\\dagger V_l = 0, \\qquad V_l^\\dagger V_l = 1.
```

The finite-window canonical forms, gauge-center movement, contractions, and
normalization are delegated to `WindowMPS`/`FiniteMPS`.  `momentum` stores the
Bloch phase; it cannot be inferred from the finite window and is needed by the
later translated-window environment engine.

Each window contains at least two sites (`V_l` followed by at least one `X`
tensor).  All windows in one `MomentumMPS` share the same Bloch momentum.
Initial-state construction (A14-A17), MPO environments, and non-trivial
auxiliary charge sectors are separate layers.
"""

struct MomentumMPS{W <: AbstractVector{<:WindowMPS}, Q <: Number}
    """Finite momentum-window representatives and their infinite backgrounds."""
    windowmps::W
    """Per-site crystal momentum in radians."""
    momentum::Q

    # There is intentionally no unchecked default constructor.  A q-MPS is
    # only meaningful in this representation when its first site is the
    # left-null isometry V_l of the background A_l.
    function MomentumMPS(
            windowmps::W,
            momentum::Q;
            atol::Real = 1e-10,
            rtol::Real = 1e-8,
        ) where {W <: AbstractVector, Q <: Number}
        _check_momentum(momentum)
        _check_window_collection(windowmps)
        ϕ = new{W, Q}(windowmps, momentum)
        _check_momentum_window(ϕ; atol, rtol)
        return ϕ
    end
end

function _check_momentum(momentum::Number)
    isreal(momentum) || throw(ArgumentError("MomentumMPS: momentum must be real"))
    return momentum
end

function _check_window_length(windowmps::WindowMPS)
    length(windowmps) >= 2 || throw(ArgumentError(
        "MomentumMPS: the window must contain V_l and at least one X tensor"
    ))
    return windowmps
end

function _check_window_collection(windowmps::AbstractVector)
    isempty(windowmps) && throw(ArgumentError(
        "MomentumMPS: at least one WindowMPS is required"
    ))
    ncell = nothing
    all(windowmps) do window
        window isa WindowMPS || throw(ArgumentError(
            "MomentumMPS: windowmps must contain only WindowMPS objects"
        ))
        _check_window_length(window)
        ncell === nothing && (ncell = length(window.left_gs))
        length(window.left_gs) == ncell || throw(ArgumentError(
            "MomentumMPS: all windows must use the same background unit-cell length"
        ))
        length(window.right_gs) == ncell || throw(ArgumentError(
            "MomentumMPS: left and right backgrounds must have the same unit-cell length"
        ))
        true
    end
    length(windowmps) == ncell || throw(ArgumentError(
        "MomentumMPS: one WindowMPS is required for every site in the unit cell"
    ))
    return windowmps
end

_tensor_residual(t) = norm(t)

function _check_momentum_window(
        ϕ::MomentumMPS;
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    _check_window_collection(ϕ.windowmps)
    for windowmps in ϕ.windowmps
        Al = windowmps.left_gs.AL[1]
        Vl = windowmps.AL[1]

        orthogonality = _tensor_residual(Al' * Vl)
        isometry = _tensor_residual(Vl' * Vl - id(right_virtualspace(windowmps, 1)))
        scale_orth = max(norm(Al) * norm(Vl), 1)
        scale_iso = max(norm(Vl)^2, 1)

        orthogonality <= atol + rtol * scale_orth || throw(ArgumentError(
            "MomentumMPS: a first window tensor is not orthogonal to A_l"
        ))
        isometry <= atol + rtol * scale_iso || throw(ArgumentError(
            "MomentumMPS: a first window tensor is not an isometry"
        ))
    end
    return true
end

"""
    check_momentum_mps(ϕ; atol=1e-10, rtol=1e-8)

Check the two defining properties of the first site of every window.  The
check is made on `windowmps.AL[1]`, not `windowmps.AC[1]`: the former is the
left-isometric canonical tensor, while the latter is a center-gauged tensor
and is generally not an isometry.
"""
function check_momentum_mps(
        ϕ::MomentumMPS;
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    return _check_momentum_window(ϕ; atol, rtol)
end

"""
    MomentumMPS(windowmps::WindowMPS; momentum=0)

Wrap an already factorized q-MPS window as a one-element window collection.
The first tensor must be the `V_l` site, followed by one or more ordinary MPS
tensors.  MPSKit's window and finite MPS constructors are responsible for
matching the boundary spaces.
"""
function MomentumMPS(
        windowmps::WindowMPS;
        momentum::Number = 0,
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    return MomentumMPS([windowmps], momentum; atol, rtol)
end

function MomentumMPS(
        windowmps::WindowMPS,
        momentum::Number;
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    return MomentumMPS([windowmps], momentum; atol, rtol)
end

function MomentumMPS(
        windowmps::AbstractVector{<:WindowMPS};
        momentum::Number = 0,
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    return MomentumMPS(windowmps, momentum; atol, rtol)
end

"""
    MomentumMPS(left_gs, windowmps::FiniteMPS, right_gs=left_gs; momentum=0)
    MomentumMPS(left_gs, tensors::AbstractVector, right_gs=left_gs; momentum=0)

Construct a q-MPS by passing the factorized window tensors explicitly.  The
first tensor in `tensors` must be `V_l`; the following tensors are `X_1`,
`X_2`, ... .  `WindowMPS` performs the usual finite-MPS canonical bookkeeping.
"""
function MomentumMPS(
        left_gs::InfiniteMPS,
        windowmps::FiniteMPS,
        right_gs::InfiniteMPS = left_gs;
        momentum::Number = 0,
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    return MomentumMPS(
        [WindowMPS(left_gs, windowmps, right_gs)]; momentum, atol, rtol
    )
end

function MomentumMPS(
        left_gs::InfiniteMPS,
        tensors::AbstractVector,
        right_gs::InfiniteMPS = left_gs;
        momentum::Number = 0,
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    return MomentumMPS(
        [WindowMPS(left_gs, tensors, right_gs)]; momentum, atol, rtol
    )
end

"""
    _momentum_hole(L, ket, R)

Contract a ket site tensor through the left and right bond environments.  The
result has the same three legs as `ket` and is the gradient with respect to the
conjugate of the corresponding bra site tensor.  Keeping this contraction
local makes the A16 terms readable and lets the transfer-map solve below act
only on a bond tensor.
"""
function _momentum_hole(L, ket, R)
    @plansor G[-1 -2; -3] := L[-1; 1] * ket[1 -2; 2] * R[2; -3]
    return G
end

"""
    _initialize_from_source(left_gs, B, right_gs, q; kwargs...)

Project the local A14 source tensor `B` onto the two-site q-MPS window
`[V_l, X_1]`.  The source is anchored at site 1 of the (possibly multi-site)
periodic background.  The first tensor to the right of the source is taken
from `right_gs.AR[2]`; the remaining periodic tensors are handled explicitly
or by a full-cell transfer solve.  Neither tensor is obtained by applying a
full `I-I-O-Z-Z-...` MPO.

The two finite terms (`m = 0, 1`) and the right-displaced tail are contracted
as in A16.  For the tail, A17 is evaluated without forming an inverse by
solving

```math
    (I - e^{-iq}\\mathcal T)Y = P,
```

with `\\mathcal T` the mixed `A_r`/`A_l` transfer map and `P` the bond
environment after the two-site source window.  The left-displaced terms vanish
because `V_l' * A_l = 0`.

The current initializer uses a two-site q-MPS window for every sublattice.
The source translations are separated by complete background unit cells, so
the phase in this routine is the phase per cell (`qcell`).
"""
function _initialize_from_source(
        left::InfiniteMPS,
        B::AbstractTensorMap,
        right::InfiniteMPS,
        qcell::Number;
        phase::Number = one(qcell),
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    isreal(qcell) || throw(ArgumentError("qmps: cell momentum must be real"))
    length(left) == length(right) || throw(ArgumentError(
        "qmps: left and right backgrounds must have the same unit-cell length"
    ))
    length(left) >= 1 || throw(ArgumentError("qmps: empty backgrounds are not supported"))

    ncell = length(left)
    z = cis(-qcell)
    isapprox(z, one(z); atol = atol, rtol = rtol) && throw(ArgumentError(
        "qmps: the momentum per unit cell is zero modulo 2π; subtract the " *
        "ground-state expectation (connected source) first"
    ))

    Al = left.AL[1]
    Vl = left_null(Al)

    # The left and right fixed-point bond maps are identities in the canonical
    # gauges.  The right environment is taken at the end of the two-site
    # window, not necessarily at site 1 of a multi-site unit cell.
    Lidentity = l_LL(left, 1)
    Ridentity = r_RR(right, 2)
    next_site = mod1(2, ncell)

    # m = 0: B and [V_l, X_1] start at the same site.
    L0 = transfer_left(Lidentity, B, Vl)
    G0 = _momentum_hole(L0, right.AR[next_site], Ridentity)

    # m = 1 is the explicit nearest-cell displacement term in A16.  Between
    # the source and V_l there are ncell - 1 ordinary tensors; these are the
    # corresponding right-background tensors paired with the left background.
    P1 = transfer_left(Lidentity, B, Al)
    for i in 2:ncell
        P1 = transfer_left(P1, right.AR[i], left.AL[i])
    end
    L1 = transfer_left(P1, right.AR[1], Vl)
    G1 = _momentum_hole(L1, right.AR[next_site], Ridentity)

    # m >= 2: after B, consume the remainder of its cell and one complete
    # mixed cell.  The full-cell transfer map then generates the remaining
    # translations, while the final site-1 tensor is paired with V_l.
    P = transfer_left(Lidentity, B, Al)
    for i in 2:ncell
        P = transfer_left(P, right.AR[i], left.AL[i])
    end
    Ptail = P
    for i in 1:ncell
        Ptail = transfer_left(Ptail, right.AR[i], left.AL[i])
    end
    Tleft = flip(TransferMatrix(right.AR, left.AL))
    solver = GMRES(; tol = max(atol, rtol * norm(Ptail)))
    Y, info = linsolve(Tleft, Ptail, Ptail, solver, 1, -z)
    info.converged == 1 || throw(ArgumentError(
        "qmps: A17 transfer-map solve did not converge " *
        "(residual = $(info.normres))"
    ))
    Ltail = transfer_left(Y, right.AR[1], Vl)
    Gtail = _momentum_hole(Ltail, right.AR[next_site], Ridentity)

    # The phase belongs to the bra window displacement: m=0, 1, and m>=2.
    X1 = phase * (G0 + z * G1 + z^2 * Gtail)

    # WindowMPS requires one tensor storage type for the finite and infinite
    # parts.  A real background with non-real q therefore gets a complex copy;
    # this changes only the scalar storage, not the represented background.
    left_out, right_out, V_out, X_out = left, right, Vl, X1
    if scalartype(X1) != scalartype(Al)
        left_out = complex(left)
        right_out = left === right ? left_out : complex(right)
        V_out = complex(Vl)
        X_out = complex(X1)
    end

    # Build the finite block explicitly first.  `WindowMPS(left, tensors,
    # right)` canonicalizes the tensor vector before checking the boundary;
    # with a charged right bond that intermediate QR step can choose a
    # reduced boundary space.  Constructing the `FiniteMPS` first preserves
    # the source/q-MPS boundary spaces and lets WindowMPS perform only its
    # embedding check.
    finite_window = FiniteMPS([V_out, X_out])
    return WindowMPS(left_out, finite_window, right_out)
end

"""
    _apply_local_operator(A, O)

Apply a one-site operator to one MPS tensor.  The operation is intentionally
local: the utility legs are added only to make a one-site `FiniteMPO`, and
MPSKit performs the physical contraction and virtual-leg fusion.  In
particular, this helper does not construct the semi-infinite `I-I-O-Z-Z-...`
operator.
"""
function _apply_local_operator(A::AbstractTensorMap, O::AbstractTensorMap)
    ncod = length(codomain(O))
    ndom = length(domain(O))
    pspace = physicalspace(A)

    if (ncod, ndom) == (1, 1)
        domain(O)[1] == pspace || throw(SpaceMismatch(
            "qmps: operator physical space does not match the ground state"
        ))
        local_mpo = FiniteMPO([add_util_leg(O)])
    elseif (ncod, ndom) == (1, 2)
        domain(O)[1] == pspace || throw(SpaceMismatch(
            "qmps: operator physical space does not match the ground state"
        ))
        # A (1,2) operator carries its charged leg to the right.  The output
        # bond therefore contains that leg after MPSKit fuses the MPO and MPS.
        local_mpo = FiniteMPO([add_single_util_leg(O)])
    elseif (ncod, ndom) == (2, 1)
        throw(ArgumentError(
            "qmps: a (2,1) operator carries its charged leg to the left; " *
            "the right-string initializer only supports (1,1) and right-going (1,2) operators"
        ))
    else
        throw(ArgumentError(
            "qmps: expected a one-site (1,1), (1,2), or (2,1) operator"
        ))
    end

    return (local_mpo * FiniteMPS([A])).AC[1]
end

"""
    _right_background(gs, O)

Construct the infinite state to the right of a source.  For a neutral
operator this is just `gs`.  For a right-going charged operator, build the
periodic `Z` MPO and let MPSKit evaluate `Zmpo * gs`; this is where all
site-by-site contraction and virtual-space fusion belongs.
"""
function _right_background(gs::InfiniteMPS, O::AbstractTensorMap)
    ncod = length(codomain(O))
    ndom = length(domain(O))
    if (ncod, ndom) == (1, 1)
        return gs
    elseif (ncod, ndom) == (1, 2)
        # `fZ` materializes the same braiding channel used by chargedMPO;
        # MPSKit then performs the infinite MPO-MPS fusion and canonicalization.
        Z = fZ(O)
        Zs = fill(Z, length(gs))
        return InfiniteMPO(Zs) * gs
    elseif (ncod, ndom) == (2, 1)
        throw(ArgumentError(
            "qmps: a (2,1) operator carries its charged leg to the left; " *
            "a right-going Z background is not appropriate"
        ))
    else
        throw(ArgumentError(
            "qmps: expected a one-site (1,1), (1,2), or (2,1) operator"
        ))
    end
end

"""
    qmps(gs::InfiniteMPS, O::AbstractTensorMap, q;
         atol=1e-10, rtol=1e-8)

Construct the A14 source internally from a ground-state `InfiniteMPS` and a
one-site local operator.  One `WindowMPS` is built for each site in the
ground-state unit cell.  For a right-going charged operator the right
background is `InfiniteMPO([Z, Z, ...]) * gs`, shifted together with the left
background for each sublattice; a neutral operator keeps the original
background.  The source representative at each sublattice is then projected
onto `[V_l, X_1]` by A15-A17.

The returned `MomentumMPS` always has one shared momentum and
`length(ϕ.windowmps) == length(gs)`.  The sublattice offset phase is absorbed
into the corresponding finite window so that summing the windows reconstructs
the full site-resolved Bloch source.
"""
function qmps(
        gs::InfiniteMPS,
        O::AbstractTensorMap,
        q::Number;
        atol::Real = 1e-10,
        rtol::Real = 1e-8,
    )
    _check_momentum(q)
    ncell = length(gs)
    right_base = _right_background(gs, O)
    windows = WindowMPS[]
    qcell = q * ncell

    for shift in 0:(ncell - 1)
        left = circshift(gs, shift)
        right = circshift(right_base, shift)
        B = _apply_local_operator(left.AC[1], O)
        window = _initialize_from_source(
            left,
            B,
            right,
            qcell;
            phase = cis(q * shift),
            atol,
            rtol,
        )
        push!(windows, window)
    end

    return MomentumMPS(windows, q; atol, rtol)
end

Base.length(ϕ::MomentumMPS) = length(ϕ.windowmps)
Base.size(ϕ::MomentumMPS, args...) = size(ϕ.windowmps, args...)
Base.eltype(::Type{<:MomentumMPS{W}}) where {W} = eltype(W)
Base.checkbounds(::Type{Bool}, ϕ::MomentumMPS, i::Integer) =
    checkbounds(Bool, ϕ.windowmps, i)

physicalspace(ϕ::MomentumMPS, i::Integer, site::Integer) =
    physicalspace(ϕ.windowmps[i], site)
physicalspace(ϕ::MomentumMPS) = map(physicalspace, ϕ.windowmps)
left_virtualspace(ϕ::MomentumMPS, i::Integer, site::Integer) =
    left_virtualspace(ϕ.windowmps[i], site)
left_virtualspace(ϕ::MomentumMPS) = map(left_virtualspace, ϕ.windowmps)
right_virtualspace(ϕ::MomentumMPS, i::Integer, site::Integer) =
    right_virtualspace(ϕ.windowmps[i], site)
right_virtualspace(ϕ::MomentumMPS) = map(right_virtualspace, ϕ.windowmps)

"""Return the canonical window tensor at site `i`."""
Base.getindex(ϕ::MomentumMPS, i::Integer) = ϕ.windowmps[i]

scalartype(ϕ::MomentumMPS) = scalartype(ϕ.windowmps[1])

"""Bloch phase acquired by translating `ϕ` by `n` sites."""
translation_phase(ϕ::MomentumMPS, n::Integer) = cis(ϕ.momentum * n)

function Base.copy(ϕ::MomentumMPS)
    return MomentumMPS(map(copy, ϕ.windowmps), ϕ.momentum)
end

function dot(bra::MomentumMPS, ket::MomentumMPS)
    isapprox(cis(bra.momentum), cis(ket.momentum); atol = 1e-12, rtol = 0) ||
        throw(ArgumentError("MomentumMPS: inner products require equal momenta"))
    length(bra) == length(ket) || throw(ArgumentError(
        "MomentumMPS: inner products require equal numbers of windows"
    ))
    return sum(dot(bra.windowmps[i], ket.windowmps[i]) for i in eachindex(bra.windowmps))
end

norm(ϕ::MomentumMPS) = sqrt(abs(real(dot(ϕ, ϕ))))

function normalize!(ϕ::MomentumMPS)
    n = norm(ϕ)
    iszero(n) && throw(ArgumentError("MomentumMPS: cannot normalize a zero state"))
    rmul!.(ϕ.windowmps, inv(n))
    return ϕ
end

normalize(ϕ::MomentumMPS) = normalize!(copy(ϕ))

function rmul!(ϕ::MomentumMPS, α::Number)
    rmul!.(ϕ.windowmps, α)
    return ϕ
end

function lmul!(α::Number, ϕ::MomentumMPS)
    lmul!.(α, ϕ.windowmps)
    return ϕ
end

Base.:*(ϕ::MomentumMPS, α::Number) = rmul!(copy(ϕ), α)
Base.:*(α::Number, ϕ::MomentumMPS) = lmul!(α, copy(ϕ))
