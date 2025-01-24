# """
#     Matsubara Green function based on Exact Diagonalization.
#     G = ∑_ij e^(βE_0) <0|A_i e^(-βH) B_j|0> 
# """
# function matsubara(T, H, gs, E0, as, bs; kwags...)
#     gf = zeros(Float64, length(as), length(bs))
#     @sync @distributed for (i, b) in enumerate(bs)
#         ket, = exponentiate(H, -1/T, b*gs; kwags...)
#         for (j, a) in enumerate(as)
#             gf[i, j] = dot(gs, a*ket)
#         end
#     end
#     return -((1/T)^2)*gf
# end

# function j_l(ops, site::Integer, direction::Vector)
#     lops = filter(op -> (length(op)==2)&&((op.id[1].index.site)!==(op.id[2].index.site))&&((op.id[1].index.site==site)||(op.id[2].index.site==site)), collect(ops))
#     jops = Vector(undef, length(lops))
#     for (i, op) in enumerate(lops)
#         value = op.value
#         distance = op.id[1].rcoordinate - op.id[2].rcoordinate
#         jops[i] = value*(dot(distance, direction)/norm(direction))*op
#     end
#     return reduce(+, jops)
# end

