using TensorKit
using MPSKit
using DCorrelators
using Plots

# give filling = (a,b), where a=b indicates half-filling, a<b indicates hole-doping and a>b indicates electron-doping
filling = (1,1)
# give a hamiltonian
H = hubbard(Float64, SU2Irrep, U1Irrep; filling=filling, t=1, U=8, μ=0)
# give a N-site random initial state 
N=4
st = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling=filling)
#find the ground state |gs> 
gs,envs,delta = find_groundstate(st, H, DMRG2(trscheme= truncbelow(1e-6)));
#obtain c^†_1|gs> and c^†_4|gs> 
ep =  e_plus(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
i, j = 1, 4
cgs₁ = chargedMPS(ep, gs, i)
cgs₂ = chargedMPS(ep, gs, j)
#calculate the propagator: <gs|c_1(t)c^†_4|gs>
dt = 0.05
ft = 10
pros = propagator(H, cgs₁, cgs₂; rev=false, dt=dt, ft=ft)
title = "Im<C_$i(dt)C^†_$j(0)> step=$dt, finialtime=$ft"
f = plot(collect(0:dt:ft),-imag.(pros), title=title, legend=false)
savefig(f,"./src/example/$title.png")
