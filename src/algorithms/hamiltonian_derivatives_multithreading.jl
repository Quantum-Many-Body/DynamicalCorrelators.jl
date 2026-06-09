struct JordanMPO_AChannel_Multithreading{L,O,R}
    leftenv::L
    localop::O
    rightenv::R
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
