module DynamicalCorrelators

using LinearAlgebra: norm, inv, mul!, I, tr, dot, BLAS, logabsdet
using QuantumLattices: Hilbert, Term, Lattice, Neighbors, azimuth, rcoordinate, bonds, Bond, OperatorGenerator, Operator, CompositeIndex, CoordinatedIndex, FockIndex, Index, OperatorSet
using QuantumLattices: AbstractLattice as QLattice, Table, isintracell, OperatorIndexToTuple, icoordinate, ReciprocalSpace, issubordinate
using TensorOperations: promote_contract, AbstractBackend, DefaultBackend, DefaultAllocator
using TensorKit: FermionParity, Trivial, U1Irrep, SU2Irrep, SU2Space, Vect, Sector, ProductSector, AbstractTensorMap, TensorMap, BraidingStyle, BraidingTensor, sectortype, sectors, Bosonic
# NOTE: `numout`/`numin` must stay in scope: TensorKit's `@planar`/`@plansor` macros
# splice *unqualified* calls to them into the caller's code.
using TensorKit: truncrank, truncerror, trunctol, ←, space, numout, numin, dual, fuse, svd_trunc!, svd_compact!, normalize!,normalize, oneunit, notrunc, similarstoragetype, insertleftunit, insertrightunit, removeunit
using TensorKit: left_null, right_null!, catdomain, catcodomain, qr_compact!, left_orth, right_orth, rmul!
using TensorKit: ⊠, ⊗, permute, repartition, domain, codomain, isomorphism, isometry, storagetype, @plansor, @planar, @tensor, blocks, block, flip, dim, infimum, id, zerovector, tensormaptype
using BlockTensorKit: nonzero_pairs, nonzero_length
using MPSKit: FiniteMPS, InfiniteMPS, FiniteMPOHamiltonian, MPOHamiltonian, TDVP, TDVP2, DMRG, DMRG2, changebonds!, SvdCut, OptimalExpand, left_virtualspace, right_virtualspace
using MPSKit: add_util_leg, _firstspace, decompose_localmpo, TransferMatrix, environments, expectation_value, physicalspace
using MPSKit: spacetype, fuse_mul_mpo, fuser, MPOTensor, approximate, LAPACK_DivideAndConquer, timestep, timestep!
using MPSKit: AbstractFiniteMPS, Algorithm, MPSTensor, MPSBondTensor, check_unambiguous_braiding, scalartype
# unexported internals used by the sweep drivers in algorithms/dmrg.jl (called, not extended)
using MPSKit: local_update!, _sweep_ranges, _num_updates, default_allocator, SerialScheduler, AdaptiveKrylov
using MPSKit: leftenv, rightenv, JordanMPOTensor, JordanMPO_AC_Hamiltonian, JordanMPO_AC2_Hamiltonian, prepare_operator!!
using KrylovKit: Lanczos, ModifiedGramSchmidt
using MPSKitModels: contract_onesite, contract_twosite, @mpoham, vertices, nearest_neighbours, next_nearest_neighbours
using MPSKitModels: InfiniteChain, InfiniteCylinder, InfiniteHelix, InfiniteLadder, FiniteChain, FiniteCylinder, FiniteStrip, FiniteHelix, FiniteLadder
using MPSKitModels: AbstractLattice as MLattice, S_x, S_y
using Distributed: @sync, @distributed, workers, addprocs, @everywhere
using SharedArrays: SharedArray
using NumericalIntegration: integrate
using JLD2: save, load, jldopen, write, close, keys
using Printf: @printf, @sprintf
using Dates
using TimerOutputs: TimerOutput, @timeit

import QuantumLattices: expand
import MPSKit: FiniteMPO, dot, correlator, transfer_left, transfer_right, AC_hamiltonian, AC2_hamiltonian, DerivativeOperator
import MPSKitModels: S_plus, S_min, S_z

# ── includes ──
include("models/lattices.jl")
include("models/hamiltonians.jl")

include("operators/fermions.jl")
include("operators/spin.jl")
include("operators/chargedmpo.jl")
include("operators/operator2mpo.jl")

include("states/chargedmps.jl")
include("states/randmps.jl")

include("utility/tools.jl")

include("algorithms/dmrg.jl")
include("algorithms/hamiltonian_threaded.jl")
include("algorithms/cpt.jl")

include("utility/defaults.jl")

include("observables/correlator.jl")
include("observables/dcorrelator.jl")
include("observables/conductivity.jl")
include("observables/fourier.jl")

# ── exports ──
export CustomLattice, BilayerSquare, Square, Custom, twosite_bonds, onesite_bonds, find_position, snake_2D, kitaev_bonds
export hubbard, extended_hubbard, hubbard_bilayer_2band, kitaev_hubbard, heisenberg_model, JKGGp_model

export fZ, e_plus, e_min, hopping, cdagc, ccdag, σz_hopping, number, onsiteCoulomb, S_plus, S_min, S_z, S_square, neiborCoulomb, heisenberg, spinflip, pairhopping
export singlet_dagger, singlet, triplet_dagger, triplet
export chargedMPO, identityMPO, hamiltonian

export FiniteNormalMPS, FiniteSuperMPS, chargedMPS, identityMPS, randFiniteMPS, randInfiniteMPS

export add_single_util_leg, cart2polar, phase_by_polar, sort_by_distance, transfer_left, contract_MPO
export myDMRG1, myDMRG2, myTDVP1, myTDVP1_CBE, myTDVP2, myDMRG1_CBE
export dmrg1, dmrg1!, dmrg2, dmrg2!
export dmrg_mix, dmrg_mix!
export set_threaded_hamiltonian!
export Perioder, CPT, singleParticleGreenFunction, spectrum, densityofstates, GrandPotential, OrderParameters

export AbstractCorrelation, PairCorrelation, pair_amplitude_indices, TwoSiteCorrelation, OneSiteCorrelation, site_indices, correlator
export evolve_mps, dcorrelator, sweep_dot
export conductivity
export fourier_kw, fourier_rw, fourier_rz, fourier_riw, static_structure_factor


function __init__()
    BLAS.set_num_threads(1)
end

end #module
