########################################################################################
using MKL
using LinearAlgebra
BLAS.set_num_threads(1)
using Pkg
Pkg.activate(".")
using TensorKit
using MPSKit
using OhMyThreads
using MPSKitModels: contract_onesite, contract_twosite
using DynamicalCorrelators
using JLD2: save,load

TensorKit.with_blockscheduler(DynamicScheduler()) do
    TensorKit.with_subblockscheduler(DynamicScheduler()) do

    elt = Float64
    filling = (3, 4)
    L = 16
    H = hubbard_bilayer_2band(elt, SU2Irrep, U1Irrep, BilayerSquare(2, 2; norbit=2); 
    tzz20 = 0, txx20 = 0, tzz2z = 0, txx1z = 0, txx2z = 0, txz2z = 0, U = 3.6, Up = 2.5, J = -0.6*2, J2 = 0.6, UpJ2 = 2.5 - 0.6/2,
    filling=filling)
    st = randFiniteMPS(elt, SU2Irrep, U1Irrep, L; md=50, filling=filling)
    D = 2048
    gs, envs, delta = find_groundstate(st, H, DMRG2(trscheme = truncrank(D) & truncerror(;rtol=1e-6), maxiter=3, verbosity=4));
    println(expectation_value(gs, H))
    println(delta)
    save("...jld2", "gs", gs, "delta", delta, "dim", D)

    end
end
########################################################################################
id = parse(Int, ARGS[1])

filling=(3, 4)

elt = Float64
gs = load("...", "gs")
gs = complex(gs)

H = hubbard_bilayer_2band(elt, SU2Irrep, U1Irrep, BilayerSquare(2, 2; norbit=2); 
tzz20 = 0, txx20 = 0, tzz2z = 0, txx1z = 0, txx2z = 0, txz2z = 0,
filling=filling)

cp = e_plus(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)
cm = e_min(Float64, SU2Irrep, U1Irrep; side=:L, filling=filling)

mps = [[chargedMPS(cp, gs, i) for i in 1:length(gs)]; [chargedMPS(cm, gs, i) for i in 1:length(gs)]]

dt = 0.06
ft = 24
n = 2
err = 1e-3
trscheme=truncerror(;rtol=err)

half = length(mps)÷2

TensorKit.with_blockscheduler(DynamicScheduler()) do
    TensorKit.with_subblockscheduler(DynamicScheduler()) do
        if id <= half
            gf = propagator(H, mps[1:half], mps[id]; rev=false, dt=dt, ft=ft, n=n, trscheme=trscheme) 
        else
            gf = propagator(H, mps[(half+1):end], mps[id]; rev=true, dt=dt, ft=ft, n=n, trscheme=trscheme)
        end
        save(".../gf_slice_$(id)_$(dt)_$(ft).jld2", "gf", gf, "index", id, "err", err)
        
    end
end
########################################################################################
TensorKit.with_blockscheduler(DynamicScheduler()) do
    TensorKit.with_subblockscheduler(DynamicScheduler()) do
        
    st = load("...jld2", "gs")
    H = hubbard_bilayer_2band(elt, SU2Irrep, U1Irrep, BilayerSquare(2, 2; norbit=2); 
    tzz20 = 0, txx20 = 0, tzz2z = 0, txx1z = 0, txx2z = 0, txz2z = 0,
    filling=filling)
    D = 4096
    gs, envs, delta = find_groundstate(st, H, DMRG2(trscheme = truncrank(D) & truncerror(;rtol=1e-8), maxiter=5, verbosity=4));
    println(expectation_value(gs, H))
    println(delta)
    save("...jld2", "gs", gs, "delta", delta, "dim", D)

    end
end
########################################################################################
using TensorKit
using MPSKit
using DynamicalCorrelators
using JLD2
gfs = [load(".../gf_slice_$(i)_$(dt)_$(ft).jld2", "gf") for i in 1:32]

ts = 0:0.06:24

gfxt = dcorrelator(RetardedGF{:f}, gfs, E0, ts)
save(".../gf_xt.jld2", "gfxt", gfxt, "ts", ts)

ws = range(11.5, 16, 300)
gfrw = fourier_rw(gfxt, ts, ws; eta=0.08)
save(".../gf_rw.jld2", "gfrw", gfrw, "ws", ws)
########################################################################################
using Pkg
Pkg.activate(".")
using QuantumLattices
using ExactDiagonalization
using QuantumClusterTheories
using Plots
function intralayer(bond::Bond)
    if bond.points[1].rcoordinate[3] == bond.points[2].rcoordinate[3]
        return 1.0
    else
        return 0.0
    end
end
function interlayer(bond::Bond)
    if bond.points[1].rcoordinate[3] == bond.points[2].rcoordinate[3]
        return 0.0
    else
        return 1.0
    end
end

function intradifforbits(bond::Bond)
    if bond.points[1].rcoordinate[3] == bond.points[2].rcoordinate[3]
        if (bond.points[1].rcoordinate[2] - bond.points[2].rcoordinate[2]) ≈ 0
            return 1.0
        elseif (bond.points[1].rcoordinate[1] - bond.points[2].rcoordinate[1]) ≈ 0
            return -1.0
        else
            return 0.0
        end
    else
        return 0.0
    end
end
function interdifforbits(bond::Bond)
    if bond.points[1].rcoordinate[3] !== bond.points[2].rcoordinate[3]
        if (bond.points[1].rcoordinate[2] - bond.points[2].rcoordinate[2]) ≈ 0
            return -1.0
        elseif (bond.points[1].rcoordinate[1] - bond.points[2].rcoordinate[1]) ≈ 0
            return 1.0
        else
            return 0.0
        end
    else
        return 0.0
    end
end

tz10 = Hopping(:tz10, -0.1123, 1, Coupling(Index(:, FID(1, :, :)), Index(:, FID(1, :, :))); amplitude=intralayer)
tz20 = Hopping(:tz20, -0.0142, 2, Coupling(Index(:, FID(1, :, :)), Index(:, FID(1, :, :))); amplitude=intralayer)
tz1z = Hopping(:tz1z, -0.6420, 1, Coupling(Index(:, FID(1, :, :)), Index(:, FID(1, :, :))); amplitude=interlayer)
tz2z = Hopping(:tz2z,  0.0257, 2, Coupling(Index(:, FID(1, :, :)), Index(:, FID(1, :, :))); amplitude=interlayer)

tx10 = Hopping(:tx10, -0.4897, 1, Coupling(Index(:, FID(2, :, :)), Index(:, FID(2, :, :))); amplitude=intralayer)
tx20 = Hopping(:tx20,  0.0686, 2, Coupling(Index(:, FID(2, :, :)), Index(:, FID(2, :, :))); amplitude=intralayer)
tx1z = Hopping(:tx1z,  0.0029, 1, Coupling(Index(:, FID(2, :, :)), Index(:, FID(2, :, :))); amplitude=interlayer)
tx2z = Hopping(:tx2z,  0.0006, 2, Coupling(Index(:, FID(2, :, :)), Index(:, FID(2, :, :))); amplitude=interlayer)

txz10_12 = Hopping(:txz10_12, 0.2425, 1, Coupling(Index(:, FID(1, :, :)), Index(:, FID(2, :, :))); amplitude=intradifforbits)
txz10_21 = Hopping(:txz10_21, 0.2425, 1, Coupling(Index(:, FID(2, :, :)), Index(:, FID(1, :, :))); amplitude=intradifforbits)
txz2z_12 = Hopping(:txz2z_12, 0.0370, 2, Coupling(Index(:, FID(1, :, :)), Index(:, FID(2, :, :))); amplitude=interdifforbits)
txz2z_21 = Hopping(:txz2z_21, 0.0370, 2, Coupling(Index(:, FID(2, :, :)), Index(:, FID(1, :, :))); amplitude=interdifforbits)


muz = Onsite(:mux, 10.5124, Coupling(Index(:, FID(1, :, :)), Index(:, FID(1, :, :))))
mux = Onsite(:muz, 10.8716, Coupling(Index(:, FID(2, :, :)), Index(:, FID(2, :, :))))

unitcell = Lattice([0, 0, 0], [0, 0, -1]; vectors = [[1, 0, 0], [0, 1, 0], [0, 0, -100]], name=:bl)
lattice = Lattice(unitcell, (2,2,1), ('p','p','o'))
hilbert = Hilbert(site=>Fock{:f}(2, 1) for site=1:length(lattice))

origiterms = (tz10, tz1z, tx10, txz10_12, txz10_21, muz, mux)
referterms = (tz10, tz1z, tx10, txz10_12, txz10_21, muz, mux)
using JLD2: load as jldload, save as jldsave
gfrw = jldload("data/gf_rw.jld2", "gfrw")
vca = VCA(:N, unitcell, lattice, hilbert, origiterms, referterms, gfrw/2)

k_path = ReciprocalPath(reciprocals([[1, 0, 0], [0, 1, 0]]),rectangle"Γ-X-M-Γ", length=200)
ω_range = jldload("data/gf_rw.jld2", "ws")
@time G = singleParticleGreenFunction(:f, vca, k_path, ω_range)
A = spectrum(G)
using Plots
f = plot(k_path, ω_range, A; ylims=(11.5, 15), clims=(0, 5), xlabel="k", ylabel="ω", title="Spectral Function")