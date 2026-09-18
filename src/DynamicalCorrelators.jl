module DynamicalCorrelators

using LinearAlgebra: norm, inv, mul!, I, tr, dot, BLAS, logabsdet
using QuantumLattices: Hilbert, Term, Lattice, Neighbors, azimuth, rcoordinate, bonds, Bond, OperatorGenerator, Operator, CompositeIndex, CoordinatedIndex, FockIndex, Index, OperatorSet
using QuantumLattices: AbstractLattice as QLattice, Table, isintracell, OperatorIndexToTuple, icoordinate, ReciprocalSpace, issubordinate
using TensorOperations: promote_contract, AbstractBackend, DefaultBackend, DefaultAllocator
using TensorKit: FermionParity, Trivial, U1Irrep, SU2Irrep, SU2Space, Vect, Sector, ProductSector, AbstractTensorMap, TensorMap, BraidingStyle, BraidingTensor, sectortype, sectors, Bosonic
using TensorKit: truncrank, truncerror, trunctol, ←, space, numout, numin, dual, fuse, svd_trunc, svd_trunc!, svd_compact!, normalize!,normalize, oneunit, notrunc, similarstoragetype, insertleftunit, insertrightunit, removeunit
using TensorKit: left_null, right_null!, catdomain, catcodomain, qr_compact!, left_orth, right_orth, lmul!, rmul!
using TensorKit: ⊠, ⊗, permute, repartition, domain, codomain, isomorphism, isometry, storagetype, @plansor, @planar, @tensor, blocks, block, flip, dim, infimum, id, zerovector, zerovector!, tensormaptype
using TensorKit: diagview, ProductSpace, set_num_transformer_threads, set_num_manipulation_threads, get_num_transformer_threads, get_num_manipulation_threads
using BlockTensorKit: nonzero_pairs, nonzero_length, SumSpace, BlockTensorMap
using MatrixAlgebraKit: TruncatedAlgorithm, initialize_output, truncate, truncation_error, SafeDivideAndConquer
using SerializedElementArrays: SerializedElementArray, filename, disk as serialize_disk
using MPSKit: FiniteMPS, InfiniteMPS, FiniteMPOHamiltonian, MPOHamiltonian, TDVP, TDVP2, DMRG, DMRG2, changebonds!, SvdCut, OptimalExpand, left_virtualspace, right_virtualspace
using MPSKit: add_util_leg, _firstspace, decompose_localmpo, TransferMatrix, environments, expectation_value, physicalspace
using MPSKit: FiniteEnvironments
using MPSKit: spacetype, fuse_mul_mpo, fuser, MPOTensor, timestep, timestep!
using MPSKit: AbstractFiniteMPS, Algorithm, MPSTensor, MPSBondTensor, check_unambiguous_braiding, scalartype
using MPSKit: default_allocator, SerialScheduler, AdaptiveKrylov, adapt_solver
using MPSKit: JordanMPOTensor, JordanMPO_AC_Hamiltonian, JordanMPO_AC2_Hamiltonian, prepare_operator!!
using MPSKit: site_type, calc_galerkin, changebond!, AC2, project_complement!, fixedpoint, gauge!, gauge2!, _transpose_tail, _transpose_front
using MPSKit: AC_projection, AC2_projection, inner_alg_gauge, Zipup, approximate
using MPSKit: C_hamiltonian
using MPSKit: integrate as mpskit_integrate
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
import MPSKit: FiniteMPO, dot, correlator, transfer_left, transfer_right, AC_hamiltonian, AC2_hamiltonian, DerivativeOperator, local_update!
import MPSKit: leftenv, rightenv
import Base: length
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

include("finiteengine/config.jl")
include("finiteengine/environments.jl")
include("finiteengine/factorizations.jl")
include("finiteengine/localupdate.jl")
include("finiteengine/timeevolution.jl")
include("finiteengine/approximate.jl")

include("algorithms/dmrg.jl")
include("algorithms/hamiltonian_threaded.jl")
include("algorithms/cpt.jl")

include("utility/defaults.jl")

include("observables/correlator.jl")
include("observables/dcorrelator.jl")
include("observables/conductivity.jl")
include("observables/fourier.jl")

# ── exports ──
export CustomLattice, BilayerSquare, Square, Custom, twosite_bonds, onesite_bonds, find_position, snake_2D, regroup_by_basis, kitaev_bonds
export hubbard, extended_hubbard, hubbard_bilayer_2band, kitaev_hubbard, heisenberg_model, JKGGp_model

export fZ, e_plus, e_min, hopping, cdagc, ccdag, σz_hopping, number, onsiteCoulomb, S_plus, S_min, S_z, S_square, neiborCoulomb, heisenberg, spinflip, pairhopping
export singlet_dagger, singlet, triplet_dagger, triplet
export chargedMPO, identityMPO, hamiltonian

export FiniteNormalMPS, FiniteSuperMPS, chargedMPS, chargedMPS!, identityMPS, randFiniteMPS, randInfiniteMPS

export add_single_util_leg, cart2polar, phase_by_polar, sort_by_distance, transfer_left, contract_MPO
export myDMRG1, myDMRG2, myTDVP1, myTDVP1_CBE, myTDVP2, myDMRG1_CBE
export dmrg1, dmrg1!, dmrg2, dmrg2!
export dmrg_mix, dmrg_mix!
export set_threaded_hamiltonian!, set_prefused_hamiltonian!
export HalfFiniteEnvironments, free_left!, free_right!, env_memory_bytes, configure_finite_engine!
export fast_timestep!, fast_approximate!
export Perioder, CPT, singleParticleGreenFunction, spectrum, densityofstates, GrandPotential, OrderParameters

export AbstractCorrelation, PairCorrelation, pair_amplitude_indices, TwoSiteCorrelation, OneSiteCorrelation, site_indices, correlator
export evolve_mps, dcorrelator, sweep_dot
export conductivity
export fourier_kw, fourier_rw, fourier_rz, fourier_riw, static_structure_factor


function __init__()
    BLAS.set_num_threads(1)
end

end #module
