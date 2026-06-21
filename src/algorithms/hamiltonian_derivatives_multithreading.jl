"""
Sparse nonzero `A`-block channel used by the multithreaded Jordan-MPO
one-site effective Hamiltonian.
"""
struct JordanMPO_AChannel_Multithreading{L,O,R}
    leftenv::L
    localop::O
    rightenv::R
end

"""
Sparse nonzero `A-A` channel used by multithreaded two-site effective
Hamiltonians and direct CBE projection.
"""
struct JordanMPO_AAChannel_Multithreading{L,O1,O2,R}
    leftenv::L
    localop1::O1
    localop2::O2
    rightenv::R
end

"""
Sparse `C-A` channel contribution used by the direct CBE projector.
"""
struct JordanMPO_CBE_CAChannel_Multithreading{O1,O2,R}
    localop1::O1
    localop2::O2
    rightenv::R
end

"""
Sparse `A-B` channel contribution used by the direct CBE projector.
"""
struct JordanMPO_CBE_ABChannel_Multithreading{L,O1,O2}
    leftenv::L
    localop1::O1
    localop2::O2
end

"""
Sparse `C-B` channel contribution used by the direct CBE projector.
"""
struct JordanMPO_CBE_CBChannel_Multithreading{O1,O2}
    localop1::O1
    localop2::O2
end

function _collect_valid_A_channels(GL, A, GR)
    nchannels = nonzero_length(A)
    nchannels == 0 && return JordanMPO_AChannel_Multithreading[]

    pairs = nonzero_pairs(A)
    first = iterate(pairs)
    first === nothing && return JordanMPO_AChannel_Multithreading[]

    ((I, Aij), state) = first
    first_channel = JordanMPO_AChannel_Multithreading(GL[I[1]], Aij, GR[I[4]])
    valid_A_channels = Vector{typeof(first_channel)}(undef, nchannels)
    valid_A_channels[1] = first_channel

    channel_index = 2
    while channel_index <= nchannels
        next = iterate(pairs, state)
        next === nothing && break

        ((I, Aij), state) = next
        valid_A_channels[channel_index] = JordanMPO_AChannel_Multithreading(
            GL[I[1]], Aij, GR[I[4]]
        )
        channel_index += 1
    end

    if channel_index <= nchannels
        resize!(valid_A_channels, channel_index - 1)
    end

    return valid_A_channels
end

function _apply_A_channels_threaded(valid_A_channels, x)
    isempty(valid_A_channels) && return zerovector(x)

    Hx = Ref{Any}(nothing)
    lk = ReentrantLock()
    idx = Threads.Atomic{Int}(1)
    nt = min(Threads.nthreads(), length(valid_A_channels))

    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            while true
                idx_t = Threads.atomic_add!(idx, 1)
                idx_t > length(valid_A_channels) && break

                channel = valid_A_channels[idx_t]
                tmp = @plansor tmp[-1 -2; -3] :=
                    channel.leftenv[-1 5; 4] *
                    x[4 2; 1] *
                    channel.localop[5 -2; 2 3] *
                    channel.rightenv[1 3; -3]

                lock(lk)
                try
                    if Hx[] === nothing
                        Hx[] = tmp
                    else
                        Hx[] = Hx[] + tmp
                    end
                finally
                    unlock(lk)
                end
            end
        end
    end

    return Hx[] === nothing ? zerovector(x) : Hx[]
end

function _collect_valid_AA_channels(GL, A1, A2, GR)
    n1 = nonzero_length(A1)
    n1 == 0 && return JordanMPO_AAChannel_Multithreading[]
    n2 = nonzero_length(A2)
    n2 == 0 && return JordanMPO_AAChannel_Multithreading[]

    first_channel = nothing
    for (I1, A1ij) in nonzero_pairs(A1), (I2, A2ij) in nonzero_pairs(A2)
        I1[4] == I2[1] || continue
        first_channel = JordanMPO_AAChannel_Multithreading(
            GL[I1[1]], A1ij, A2ij, GR[I2[4]]
        )
        break
    end
    first_channel === nothing && return JordanMPO_AAChannel_Multithreading[]

    valid_AA_channels = Vector{typeof(first_channel)}()
    sizehint!(valid_AA_channels, n1 * n2)
    for (I1, A1ij) in nonzero_pairs(A1), (I2, A2ij) in nonzero_pairs(A2)
        I1[4] == I2[1] || continue
        push!(
            valid_AA_channels,
            JordanMPO_AAChannel_Multithreading(GL[I1[1]], A1ij, A2ij, GR[I2[4]]),
        )
    end

    return valid_AA_channels
end

function _apply_AA_channels_threaded(valid_AA_channels, x)
    isempty(valid_AA_channels) && return zerovector(x)

    Hx = Ref{Any}(nothing)
    lk = ReentrantLock()
    idx = Threads.Atomic{Int}(1)
    nt = min(Threads.nthreads(), length(valid_AA_channels))

    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            while true
                idx_t = Threads.atomic_add!(idx, 1)
                idx_t > length(valid_AA_channels) && break

                channel = valid_AA_channels[idx_t]
                tmp = @plansor tmp[-1 -2; -3 -4] :=
                    channel.leftenv[-1 2; 1] *
                    x[1 3; 7 5] *
                    channel.localop1[2 -2; 3 4] *
                    channel.localop2[4 -4; 5 6] *
                    channel.rightenv[7 6; -3]

                lock(lk)
                try
                    if Hx[] === nothing
                        Hx[] = tmp
                    else
                        Hx[] = Hx[] + tmp
                    end
                finally
                    unlock(lk)
                end
            end
        end
    end

    return Hx[] === nothing ? zerovector(x) : Hx[]
end

"""
Channel-decomposed direct CBE projector for finite Jordan-MPO Hamiltonians.

The object stores the null-space projectors `NL`/`NR`, the two neighboring MPS
tensors, and sparse `AA`, `CA`, `AB`, and `CB` channel lists. Applying it
contracts each channel directly into the CBE intermediate, avoiding full
two-site materialization.
"""
struct JordanMPO_CBE_Project_Multithreading{NLT,NRT,LT,RT,AA,CA,AB,CB}
    NL::NLT
    NR::NRT
    left_tensor::LT
    right_tail::RT
    valid_AA_channels::AA
    valid_CA_channels::CA
    valid_AB_channels::AB
    valid_CB_channels::CB
end

function _collect_valid_CA_channels(C1, A2, GR)
    n1 = nonzero_length(C1)
    n1 == 0 && return JordanMPO_CBE_CAChannel_Multithreading[]
    n2 = nonzero_length(A2)
    n2 == 0 && return JordanMPO_CBE_CAChannel_Multithreading[]

    first_channel = nothing
    for (I1, C1ij) in nonzero_pairs(C1), (I2, A2ij) in nonzero_pairs(A2)
        I1[3] == I2[1] || continue
        first_channel = JordanMPO_CBE_CAChannel_Multithreading(C1ij, A2ij, GR[I2[4]])
        break
    end
    first_channel === nothing && return JordanMPO_CBE_CAChannel_Multithreading[]

    valid_CA_channels = Vector{typeof(first_channel)}()
    sizehint!(valid_CA_channels, n1 * n2)
    for (I1, C1ij) in nonzero_pairs(C1), (I2, A2ij) in nonzero_pairs(A2)
        I1[3] == I2[1] || continue
        push!(
            valid_CA_channels,
            JordanMPO_CBE_CAChannel_Multithreading(C1ij, A2ij, GR[I2[4]]),
        )
    end

    return valid_CA_channels
end

function _collect_valid_AB_channels(GL, A1, B2)
    n1 = nonzero_length(A1)
    n1 == 0 && return JordanMPO_CBE_ABChannel_Multithreading[]
    n2 = nonzero_length(B2)
    n2 == 0 && return JordanMPO_CBE_ABChannel_Multithreading[]

    first_channel = nothing
    for (I1, A1ij) in nonzero_pairs(A1), (I2, B2ij) in nonzero_pairs(B2)
        I1[4] == I2[1] || continue
        first_channel = JordanMPO_CBE_ABChannel_Multithreading(GL[I1[1]], A1ij, B2ij)
        break
    end
    first_channel === nothing && return JordanMPO_CBE_ABChannel_Multithreading[]

    valid_AB_channels = Vector{typeof(first_channel)}()
    sizehint!(valid_AB_channels, n1 * n2)
    for (I1, A1ij) in nonzero_pairs(A1), (I2, B2ij) in nonzero_pairs(B2)
        I1[4] == I2[1] || continue
        push!(
            valid_AB_channels,
            JordanMPO_CBE_ABChannel_Multithreading(GL[I1[1]], A1ij, B2ij),
        )
    end

    return valid_AB_channels
end

function _collect_valid_CB_channels(C1, B2)
    n1 = nonzero_length(C1)
    n1 == 0 && return JordanMPO_CBE_CBChannel_Multithreading[]
    n2 = nonzero_length(B2)
    n2 == 0 && return JordanMPO_CBE_CBChannel_Multithreading[]

    first_channel = nothing
    for (I1, C1ij) in nonzero_pairs(C1), (I2, B2ij) in nonzero_pairs(B2)
        I1[3] == I2[1] || continue
        first_channel = JordanMPO_CBE_CBChannel_Multithreading(C1ij, B2ij)
        break
    end
    first_channel === nothing && return JordanMPO_CBE_CBChannel_Multithreading[]

    valid_CB_channels = Vector{typeof(first_channel)}()
    sizehint!(valid_CB_channels, n1 * n2)
    for (I1, C1ij) in nonzero_pairs(C1), (I2, B2ij) in nonzero_pairs(B2)
        I1[3] == I2[1] || continue
        push!(valid_CB_channels, JordanMPO_CBE_CBChannel_Multithreading(C1ij, B2ij))
    end

    return valid_CB_channels
end

"""
    _cbe_project_multithreading(NL, NR, left_tensor, right_tail, GL, W1, W2, GR)

Build a channel-decomposed direct CBE projector for two neighboring
Jordan-MPO tensors.

This is the optimized backend used by `_cbe_direct_project` when the Hamiltonian
is an `MPOHamiltonian{<:JordanMPOTensor}`.
"""
function _cbe_project_multithreading(NL, NR, left_tensor, right_tail, GL, W1, W2, GR)
    valid_AA_channels = _collect_valid_AA_channels(
        GL[2:(end - 1)], W1.A, W2.A, GR[2:(end - 1)]
    )
    valid_CA_channels = _collect_valid_CA_channels(W1.C, W2.A, GR[2:(end - 1)])
    valid_AB_channels = _collect_valid_AB_channels(GL[2:(end - 1)], W1.A, W2.B)
    valid_CB_channels = _collect_valid_CB_channels(W1.C, W2.B)

    return JordanMPO_CBE_Project_Multithreading(
        NL, NR, left_tensor, right_tail,
        valid_AA_channels, valid_CA_channels, valid_AB_channels, valid_CB_channels
    )
end

function _cbe_project_zero(project::JordanMPO_CBE_Project_Multithreading)
    return zerovector!(similar(project.left_tensor, space(project.NL, 3) ← space(project.NR, 1)))
end

function _cbe_project_add(intermediate, contribution)
    contribution === nothing && return intermediate
    return intermediate === nothing ? contribution : intermediate + contribution
end

function _apply_cbe_channel(channel::JordanMPO_AAChannel_Multithreading, NL, NR, left_tensor, right_tail)
    tmp = @plansor tmp[-1; -2] :=
        conj(NL[4 5; -1]) *
        channel.leftenv[4 2; 1] *
        left_tensor[1 3; 11] *
        right_tail[11; 6 7] *
        channel.localop1[2 5; 3 12] *
        channel.localop2[12 10; 7 8] *
        channel.rightenv[6 8; 9] *
        conj(NR[-2; 9 10])
    return tmp
end

function _apply_cbe_channel(channel::JordanMPO_CBE_CAChannel_Multithreading, NL, NR, left_tensor, right_tail)
    tmp = @plansor tmp[-1; -2] :=
        conj(NL[1 3; -1]) *
        left_tensor[1 2; 9] *
        right_tail[9; 4 5] *
        channel.localop1[3; 2 10] *
        channel.localop2[10 8; 5 6] *
        channel.rightenv[4 6; 7] *
        conj(NR[-2; 7 8])
    return tmp
end

function _apply_cbe_channel(channel::JordanMPO_CBE_ABChannel_Multithreading, NL, NR, left_tensor, right_tail)
    tmp = @plansor tmp[-1; -2] :=
        conj(NL[4 5; -1]) *
        channel.leftenv[4 2; 1] *
        left_tensor[1 3; 9] *
        right_tail[9; 6 7] *
        channel.localop1[2 5; 3 10] *
        channel.localop2[10 8; 7] *
        conj(NR[-2; 6 8])
    return tmp
end

function _apply_cbe_channel(channel::JordanMPO_CBE_CBChannel_Multithreading, NL, NR, left_tensor, right_tail)
    tmp = @plansor tmp[-1; -2] :=
        conj(NL[1 3; -1]) *
        left_tensor[1 2; 7] *
        right_tail[7; 4 5] *
        channel.localop1[3; 2 8] *
        channel.localop2[8 6; 5] *
        conj(NR[-2; 4 6])
    return tmp
end

function _cbe_channel_at(channel_groups, idx::Int)
    for channels in channel_groups
        nchannels = length(channels)
        if idx <= nchannels
            return channels[idx]
        end
        idx -= nchannels
    end
    throw(BoundsError(channel_groups, idx))
end

function _apply_cbe_channel_groups_threaded(channel_groups, NL, NR, left_tensor, right_tail)
    nchannels_total = sum(length, channel_groups)
    nchannels_total == 0 && return nothing

    intermediate = Ref{Any}(nothing)
    lk = ReentrantLock()
    idx = Threads.Atomic{Int}(1)
    nt = min(Threads.nthreads(), nchannels_total)

    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            local_intermediate = nothing
            while true
                idx_t = Threads.atomic_add!(idx, 1)
                idx_t > nchannels_total && break

                channel = _cbe_channel_at(channel_groups, idx_t)
                tmp = _apply_cbe_channel(channel, NL, NR, left_tensor, right_tail)
                local_intermediate = _cbe_project_add(local_intermediate, tmp)
            end

            if local_intermediate !== nothing
                lock(lk)
                try
                    intermediate[] = _cbe_project_add(intermediate[], local_intermediate)
                finally
                    unlock(lk)
                end
            end
        end
    end

    return intermediate[]
end

function _apply_cbe_project(project::JordanMPO_CBE_Project_Multithreading)
    NL = project.NL
    NR = project.NR
    left_tensor = project.left_tensor
    right_tail = project.right_tail

    intermediate = _apply_cbe_channel_groups_threaded(
        (
            project.valid_AA_channels,
            project.valid_CA_channels,
            project.valid_AB_channels,
            project.valid_CB_channels,
        ),
        NL, NR, left_tensor, right_tail,
    )

    return intermediate === nothing ? _cbe_project_zero(project) : intermediate
end

"""
Multithreaded one-site effective Hamiltonian for finite Jordan-MPO operators.

This mirrors MPSKit's `JordanMPO_AC_Hamiltonian` structure but evaluates the
sparse continuing `A` block as independent nonzero channels.
"""
struct JordanMPO_AC_Hamiltonian_Multithreading{O1,O2,C} <: DerivativeOperator
    D::Union{O1,Missing}
    I::Union{O1,Missing}
    E::Union{O1,Missing}
    C::Union{O2,Missing}
    B::Union{O2,Missing}
    valid_A_channels::C
    function JordanMPO_AC_Hamiltonian_Multithreading(
            onsite, not_started, finished, starting, ending, continuing
        )
        gl = continuing[1]
        S = spacetype(gl)
        M = storagetype(gl)
        O1 = tensormaptype(S, 1, 1, M)
        O2 = tensormaptype(S, 2, 2, M)
        valid_A_channels = _collect_valid_A_channels(continuing...)
        return new{O1,O2,typeof(valid_A_channels)}(
            onsite, not_started, finished, starting, ending, valid_A_channels
        )
    end
end

function (H::JordanMPO_AC_Hamiltonian_Multithreading)(x::MPSTensor)
    y = if isempty(H.valid_A_channels)
        zerovector(x)
    else
        _apply_A_channels_threaded(H.valid_A_channels, x)
    end

    ismissing(H.D) || @plansor y[-1 -2; -3] += x[-1 1; -3] * H.D[-2; 1]
    ismissing(H.E) || @plansor y[-1 -2; -3] += H.E[-1; 1] * x[1 -2; -3]
    ismissing(H.I) || @plansor y[-1 -2; -3] += x[-1 -2; 1] * H.I[1; -3]
    ismissing(H.C) || @plansor y[-1 -2; -3] += x[-1 2; 1] * H.C[-2 -3; 2 1]
    ismissing(H.B) || @plansor y[-1 -2; -3] += H.B[-1 -2; 1 2] * x[1 2; -3]

    return y
end

"""
Multithreaded two-site effective Hamiltonian for finite Jordan-MPO operators.

The dense non-continuing terms are stored in the usual `II`, `IC`, `ID`, `CB`,
`CA`, `AB`, `BE`, `DE`, and `EE` blocks, while the sparse continuing-continuing
`AA` contribution is evaluated channel by channel across Julia threads.
"""
struct JordanMPO_AC2_Hamiltonian_Multithreading{O1,O2,O3,C} <: DerivativeOperator
    II::Union{O1,Missing}
    IC::Union{O2,Missing}
    ID::Union{O1,Missing}
    CB::Union{O2,Missing}
    CA::Union{O3,Missing}
    AB::Union{O3,Missing}
    valid_AA_channels::C
    BE::Union{O2,Missing}
    DE::Union{O1,Missing}
    EE::Union{O1,Missing}
    function JordanMPO_AC2_Hamiltonian_Multithreading(
            II, IC, ID, CB, CA, AB, continuing, BE, DE, EE
        )
        gl = continuing[1]
        S = spacetype(gl)
        M = storagetype(gl)
        O1 = tensormaptype(S, 1, 1, M)
        O2 = tensormaptype(S, 2, 2, M)
        O3 = tensormaptype(S, 3, 3, M)
        valid_AA_channels = _collect_valid_AA_channels(continuing...)
        return new{O1,O2,O3,typeof(valid_AA_channels)}(
            II, IC, ID, CB, CA, AB, valid_AA_channels, BE, DE, EE
        )
    end
end

function (H::JordanMPO_AC2_Hamiltonian_Multithreading)(x::MPOTensor)
    y = if isempty(H.valid_AA_channels)
        zerovector(x)
    else
        _apply_AA_channels_threaded(H.valid_AA_channels, x)
    end

    ismissing(H.II) || @plansor y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * H.II[-3; 1]
    ismissing(H.IC) || @plansor y[-1 -2; -3 -4] += x[-1 -2; 1 2] * H.IC[-4 -3; 2 1]
    ismissing(H.ID) || @plansor y[-1 -2; -3 -4] += x[-1 -2; -3 1] * H.ID[-4; 1]
    ismissing(H.CB) || @plansor y[-1 -2; -3 -4] += x[-1 1; -3 2] * H.CB[-2 -4; 1 2]
    ismissing(H.CA) || @plansor y[-1 -2; -3 -4] += x[-1 1; 3 2] * H.CA[-2 -4 -3; 1 2 3]
    ismissing(H.AB) || @plansor y[-1 -2; -3 -4] += x[1 2; -3 3] * H.AB[-1 -2 -4; 1 2 3]
    ismissing(H.BE) || @plansor y[-1 -2; -3 -4] += x[1 2; -3 -4] * H.BE[-1 -2; 1 2]
    ismissing(H.DE) || @plansor y[-1 -2; -3 -4] += x[-1 1; -3 -4] * H.DE[-2; 1]
    ismissing(H.EE) || @plansor y[-1 -2; -3 -4] += x[1 -2; -3 -4] * H.EE[-1; 1]

    return y
end

"""
    AC_hamiltonian(site, below, operator::MPOHamiltonian{<:JordanMPOTensor},
                   above, envs)

Construct the multithreaded one-site effective Hamiltonian for a finite
Jordan-MPO Hamiltonian.

Only the sparse continuing `A` block is split into thread-parallel channels;
the other local terms follow MPSKit's standard contraction structure.
"""
function AC_hamiltonian(
        site::Int, below::FiniteMPS{<:MPSTensor},
        operator::MPOHamiltonian{<:JordanMPOTensor}, above::FiniteMPS{<:MPSTensor}, envs
    )
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
    W = operator[site]

    # starting
    if nonzero_length(W.C) > 0
        GR_2 = GR[2:(end - 1)]
        @plansor starting_[-1 -2; -3 -4] ≔ W.C[-1; -3 1] * GR_2[-4 1; -2]
        starting = only(starting_)
    else
        starting = missing
    end

    # ending
    if nonzero_length(W.B) > 0
        GL_2 = GL[2:(end - 1)]
        @plansor ending_[-1 -2; -3 -4] ≔ GL_2[-1 1; -3] * W.B[1 -2; -4]
        ending = only(ending_)
    else
        ending = missing
    end

    # onsite
    if nonzero_length(W.D) > 0
        if !ismissing(starting)
            @plansor starting[-1 -2; -3 -4] += W.D[-1; -3] * removeunit(GR[end], 2)[-4; -2]
            onsite = missing
        elseif !ismissing(ending)
            @plansor ending[-1 -2; -3 -4] += removeunit(GL[1], 2)[-1; -3] * W.D[-2; -4]
            onsite = missing
        else
            onsite = W.D
        end
    else
        onsite = missing
    end

    # not_started
    if isfinite(operator) && site == length(operator)
        not_started = missing
    elseif !ismissing(starting)
        I = id(storagetype(GR[1]), physicalspace(W))
        @plansor starting[-1 -2; -3 -4] += I[-1; -3] * removeunit(GR[1], 2)[-4; -2]
        not_started = missing
    else
        not_started = removeunit(GR[1], 2)
    end

    # finished
    if isfinite(operator) && site == 1
        finished = missing
    elseif !ismissing(ending)
        I = id(storagetype(GL[end]), physicalspace(W))
        @plansor ending[-1 -2; -3 -4] += removeunit(GL[end], 2)[-1; -3] * I[-2; -4]
        finished = missing
    else
        finished = removeunit(GL[end], 2)
    end

    # continuing
    continuing = (GL[2:(end - 1)], W.A, GR[2:(end - 1)])

    return JordanMPO_AC_Hamiltonian_Multithreading(
        onsite, not_started, finished, starting, ending, continuing
    )
end

"""
    AC2_hamiltonian(site, below, operator::MPOHamiltonian{<:JordanMPOTensor},
                    above, envs)

Construct the multithreaded two-site effective Hamiltonian for a finite
Jordan-MPO Hamiltonian.

The continuing-continuing `AA` block is collected as nonzero channel pairs and
evaluated in parallel when the returned derivative operator is applied.
"""
function AC2_hamiltonian(
        site::Int, below::FiniteMPS{<:MPSTensor},
        operator::MPOHamiltonian{<:JordanMPOTensor}, above::FiniteMPS{<:MPSTensor}, envs
    )
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    W1 = operator[site]
    W2 = operator[site + 1]

    # starting left - continuing right
    if nonzero_length(W1.C) > 0 && nonzero_length(W2.A) > 0
        @plansor CA_[-1 -2 -3; -4 -5 -6] ≔ W1.C[-1; -4 2] * W2.A[2 -2; -5 1] *
            GR[2:(end - 1)][-6 1; -3]
        CA = only(CA_)
    else
        CA = missing
    end

    # continuing left - ending right
    if nonzero_length(W1.A) > 0 && nonzero_length(W2.B) > 0
        @plansor AB_[-1 -2 -3; -4 -5 -6] ≔ GL[2:(end - 1)][-1 2; -4] * W1.A[2 -2; -5 1] *
            W2.B[1 -3; -6]
        AB = only(AB_)
    else
        AB = missing
    end

    # middle
    if nonzero_length(W1.C) > 0 && nonzero_length(W2.B) > 0
        if !ismissing(CA)
            @plansor CA[-1 -2 -3; -4 -5 -6] += W1.C[-1; -4 1] * W2.B[1 -2; -5] *
                removeunit(GR[end], 2)[-6; -3]
            CB = missing
        elseif !ismissing(AB)
            @plansor AB[-1 -2 -3; -4 -5 -6] += removeunit(GL[1], 2)[-1; -4] *
                W1.C[-2; -5 1] * W2.B[1 -3; -6]
            CB = missing
        else
            @plansor CB_[-1 -2; -3 -4] ≔ W1.C[-1; -3 1] * W2.B[1 -2; -4]
            CB = only(CB_)
        end
    else
        CB = missing
    end

    # starting right
    if nonzero_length(W2.C) > 0
        if !ismissing(CA)
            I = id(storagetype(GR[1]), physicalspace(W1))
            @plansor CA[-1 -2 -3; -4 -5 -6] += (I[-1; -4] * W2.C[-2; -5 1]) *
                GR[2:(end - 1)][-6 1; -3]
            IC = missing
        else
            @plansor IC[-1 -2; -3 -4] ≔ W2.C[-1; -3 1] * GR[2:(end - 1)][-4 1; -2]
        end
    else
        IC = missing
    end

    # ending left
    if nonzero_length(W1.B) > 0
        if !ismissing(AB)
            I = id(storagetype(GL[end]), physicalspace(W2))
            @plansor AB[-1 -2 -3; -4 -5 -6] += GL[2:(end - 1)][-1 1; -4] *
                (W1.B[1 -2; -5] * I[-3; -6])
            BE = missing
        else
            @plansor BE[-1 -2; -3 -4] ≔ GL[2:(end - 1)][-1 2; -3] * W1.B[2 -2; -4]
        end
    else
        BE = missing
    end

    # onsite left
    if nonzero_length(W1.D) > 0
        if !ismissing(BE)
            @plansor BE[-1 -2; -3 -4] += removeunit(GL[1], 2)[-1; -3] * W1.D[-2; -4]
            DE = missing
        elseif !ismissing(AB)
            I = id(storagetype(GL[end]), physicalspace(W2))
            @plansor AB[-1 -2 -3; -4 -5 -6] += removeunit(GL[1], 2)[-1; -4] *
                (W1.D[-2; -5] * I[-3; -6])
            DE = missing
        else
            DE = only(W1.D)
        end
    else
        DE = missing
    end

    # onsite right
    if nonzero_length(W2.D) > 0
        if !ismissing(IC)
            @plansor IC[-1 -2; -3 -4] += W2.D[-1; -3] * removeunit(GR[end], 2)[-4; -2]
            ID = missing
        elseif !ismissing(CA)
            I = id(storagetype(GR[1]), physicalspace(W1))
            @plansor CA[-1 -2 -3; -4 -5 -6] += (I[-1; -4] * W2.D[-2; -5]) *
                removeunit(GR[end], 2)[-6; -3]
            ID = missing
        else
            ID = only(W2.D)
        end
    else
        ID = missing
    end

    # finished
    if isfinite(operator) && site + 1 == length(operator)
        II = missing
    elseif !ismissing(IC)
        I = id(storagetype(GR[1]), physicalspace(W2))
        @plansor IC[-1 -2; -3 -4] += I[-1; -3] * removeunit(GR[1], 2)[-4; -2]
        II = missing
    elseif !ismissing(CA)
        I = id(storagetype(GR[1]), physicalspace(W1) ⊗ physicalspace(W2))
        @plansor CA[-1 -2 -3; -4 -5 -6] += I[-1 -2; -4 -5] * removeunit(GR[1], 2)[-6; -3]
        II = missing
    else
        II = transpose(removeunit(GR[1], 2))
    end

    # unstarted
    if isfinite(operator) && site == 1
        EE = missing
    elseif !ismissing(BE)
        I = id(storagetype(GL[end]), physicalspace(W1))
        @plansor BE[-1 -2; -3 -4] += removeunit(GL[end], 2)[-1; -3] * I[-2; -4]
        EE = missing
    elseif !ismissing(AB)
        I = id(storagetype(GL[end]), physicalspace(W1) ⊗ physicalspace(W2))
        @plansor AB[-1 -2 -3; -4 -5 -6] += removeunit(GL[end], 2)[-1; -4] * I[-2 -3; -5 -6]
        EE = missing
    else
        EE = removeunit(GL[end], 2)
    end

    # continuing - continuing
    continuing = (GL[2:(end - 1)], W1.A, W2.A, GR[2:(end - 1)])

    return JordanMPO_AC2_Hamiltonian_Multithreading(
        II, IC, ID, CB, CA, AB, continuing, BE, DE, EE
    )
end
