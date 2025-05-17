"""

"""
abstract type Correlation end

struct PairingCorrelation{K, L<:CustomLattice} <: Correlation
    operator::AbstractTensorMap
    lattice:: L
    amplitude::Union{Nothing, Function}
end


struct SpinCorrelation{K} <: Correlation
    operator::AbstractTensorMap
    amplitude::Union{Nothing, Function}
end

function correlator(O::AbstractTensorMap, bra::AbstractFiniteMPS, gs::AbstractFiniteMPS, index::Integer)
    dot(bra, chargedMPS(O, gs, index))
end

function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, index::Integer, ket::AbstractFiniteMPS)
    dot(chargedMPS(O, gs, index), ket)
end


function correlator(O::AbstractTensorMap, bra::AbstractFiniteMPS, gs::AbstractFiniteMPS, index::NTuple{2, Integer})
    dot(bra, chargedMPS(O, gs, index))
end

function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, index::NTuple{2, Integer}, ket::AbstractFiniteMPS)
    dot(chargedMPS(O, gs, index), ket)
end


function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, indices::NTuple{2, Integer})
    dot(chargedMPS(O, gs, indices[1]), chargedMPS(O, gs, indices[2]))
end

function correlator(O::AbstractTensorMap, gs::AbstractFiniteMPS, indices::NTuple{4, Integer})
    dot(chargedMPS(O, gs, indices[1], indices[2]), chargedMPS(O, gs, indices[3], indices[4]))
end