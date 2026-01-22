using TensorKit
using MPSKit
using DynamicalCorrelators

# give filling = (a,b), where a=b is half-filling, a<b is h-doping and a>b is e-doping
filling = (1,1)

# give a hamiltonian
H = hubbard(Float64, SU2Irrep, U1Irrep; filling=filling, t=1, U=8, μ=0)

# give a N-site random initial state 
N=4
st = randFiniteMPS(ComplexF64, SU2Irrep, U1Irrep, N; filling=filling)

#find the ground state |gs> 
gs, envs, delta = find_groundstate(st, H, DMRG2(trscheme= trunctol(1e-6)));

#obtain c^†_1|gs> and c^†_4|gs> 
ep =  e_plus(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
i, j = 1, 4
cgs₁ = chargedMPS(ep, gs, i)
cgs₂ = chargedMPS(ep, gs, j)

#calculate the propagator: <gs|c_1(t)c^†_4|gs> (i.e. <gs|c_1(0)e^{-iHt}c^†_4|gs>)
dt = 0.05
ft = 10
pros = propagator(H, cgs₁, cgs₂; rev=false, dt=dt, ft=ft)

#calculate ground state energy
E0 = expectation_value(gs, H)

#give the creation and annihilation operators
cp =  e_plus(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
cm =  e_min(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
sp =  S_plus(Float64, SU2Irrep, U1Irrep; filling=filling)

#calculate the dynamical single-particle correlation function 
edc = dcorrelator(RetardedGF{:f}, H, E0, [[chargedMPS(cp, gs, i) for i in 1:48]; [chargedMPS(cm, gs, i) for i in 1:48]]; trscheme=truncrank(200), n=n, dt=dt, ft=ft)

#calculate the dynamical two-particle spin-spin correlation function 
sdc = dcorrelator(GreaterLessGF, H, E0, [chargedMPS(sp, gs, i) for i in 1:48]; whichs=:greater, trscheme=truncrank(200), n=n, dt=dt, ft=ft)