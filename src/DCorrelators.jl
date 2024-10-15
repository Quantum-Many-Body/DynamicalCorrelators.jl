module DCorrelators

using TensorKit
using MPSKit

export chargedMPS

function chargedMPS(state::FiniteMPS, opt::TensorMap, site::Integer)
    N = length(state)
    pspaces = [codomain(state.AL[i])[2] for i in 1:N]
    @assert (length(domain(opt))==2)&&(in(domain(opt)[1],pspaces)) "Physical and virtual space should be set at the left and right side of domain, respectively"
    As = [state.AL[i] for i in 1:N]
    As[end] = As[end] * state.CLs[end]
    @planar T[-1 -2; -3 -4] := opt[-2; 1 -4] * As[site][-1 1; -3] 
    As[site] = T * isomorphism(domain(T),fuse(domain(T)))
    for i in (site+1):N
        iso = isomorphism(fuse(codomain(As[i])[1],domain(opt)[2]),codomain(As[i])[1]*domain(opt)[2])
        @planar T[-1 -2; -4 -5] := iso[-1;2 3] * τ[3 -2; 1 -5] * As[i][2 1; -4]
        As[i] = T * isomorphism(domain(T),fuse(domain(T)))
    end
    return FiniteMPS(As)
end

end #module