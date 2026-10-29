module DynamicalCorrelators

using LinearAlgebra: inv, mul!, I, tr, dot, BLAS, logabsdet
using QuantumLattices: Hilbert, Term, Lattice, Neighbors, azimuth, rcoordinate, bonds, Bond, OperatorGenerator, Operator, CompositeIndex, CoordinatedIndex, FockIndex, Index, OperatorSet
using QuantumLattices: AbstractLattice as QLattice, Table, isintracell, OperatorIndexToTuple, icoordinate, ReciprocalSpace, issubordinate
using TensorOperations: promote_contract, AbstractBackend, DefaultBackend, DefaultAllocator, BufferAllocator
using TensorKit: FermionParity, Trivial, U1Irrep, SU2Irrep, SU2Space, Vect, Sector, ProductSector, AbstractTensorMap, TensorMap, BraidingStyle, BraidingTensor, sectortype, sectors, Bosonic, SpaceMismatch
using TensorKit: truncrank, truncerror, trunctol, ←, space, numout, numin, numind, dual, fuse, svd_trunc, svd_trunc!, svd_compact!, oneunit, notrunc, similarstoragetype, insertleftunit, insertrightunit, removeunit
using TensorKit: left_null, right_null!, catdomain, catcodomain, qr_compact!, lq_compact!, left_orth, right_orth, right_orth!, scale!, absorb!
using TensorKit: ⊠, ⊗, ⊕, permute, repartition, domain, codomain, isomorphism, isometry, storagetype, @plansor, @planar, @tensor, blocks, block, flip, dim, infimum, id, zerovector, zerovector!, tensormaptype
using TensorKit: diagview, ProductSpace, set_num_transformer_threads, set_num_manipulation_threads, get_num_transformer_threads, get_num_manipulation_threads
using BlockTensorKit: nonzero_pairs, nonzero_length, SumSpace, BlockTensorMap
using MatrixAlgebraKit: TruncatedAlgorithm, TruncationStrategy, TruncationByOrder, TruncationByError, TruncationIntersection, initialize_output, truncate, truncation_error, SafeDivideAndConquer
using SerializedElementArrays: SerializedElementArray, disk as serialize_disk
using MPSKit: FiniteMPS, InfiniteMPS, WindowMPS, InfiniteMPO, FiniteMPOHamiltonian, MPOHamiltonian, TDVP, TDVP2, DMRG, DMRG2, SvdCut
using MPSKit: add_util_leg, _firstspace, decompose_localmpo, TransferMatrix, environments, expectation_value, l_LL, r_RR
using MPSKit: FiniteEnvironments
using MPSKit: spacetype, fuse_mul_mpo, fuser, MPOTensor, timestep, timestep!
using MPSKit: AbstractFiniteMPS, Algorithm, MPSTensor, MPSBondTensor, check_unambiguous_braiding
using MPSKit: default_allocator, SerialScheduler
using MPSKit: JordanMPOTensor, JordanMPO_AC_Hamiltonian, JordanMPO_AC2_Hamiltonian, prepare_operator!!
using MPSKit: site_type, AC2, fixedpoint, gauge!, gauge2!, _transpose_tail, _transpose_front
using MPSKit: left_gauge, _lastspace
using MPSKit: AC_projection, AC2_projection, inner_alg_gauge, Zipup, approximate
using MPSKit: C_hamiltonian, OptimalExpand, project_complement!, project_complement_right!
using KrylovKit: Lanczos, ModifiedGramSchmidt, GMRES, linsolve, exponentiate
using MPSKitModels: contract_onesite, contract_twosite, @mpoham, vertices, nearest_neighbours, next_nearest_neighbours
using MPSKitModels: InfiniteChain, InfiniteCylinder, InfiniteHelix, InfiniteLadder, FiniteChain, FiniteCylinder, FiniteStrip, FiniteHelix, FiniteLadder
using MPSKitModels: AbstractLattice as MLattice, S_x, S_y
using Distributed: @sync, @distributed, workers, addprocs, @everywhere, myid, remotecall_fetch, RemoteException, ProcessExitedException, nprocs
using SharedArrays: SharedArray
using NumericalIntegration: integrate
using JLD2: save, load, jldopen, write, close, keys
using Printf: @printf, @sprintf
using Dates: canonicalize, format, now, CompoundPeriod, Day, Hour, Millisecond, Minute, Nanosecond, Period, Second, Week, value
using TimerOutputs: TimerOutput, @timeit, reset_timer!

import SerializedElementArrays: filename, pathname
import LinearAlgebra: norm
import TensorKit: normalize!, normalize, lmul!, rmul!, scalartype
import TensorKit: InnerProductStyle, EuclideanInnerProduct, throw_invalid_innerproduct, FusionStyle, UniqueFusion, blocksectors, _norm
import VectorInterface: inner, add!, add, One, promote_add, promote_inner, promote_scale, scale
import VectorInterface: scale! as vi_scale!
import VectorInterface: zerovector as vi_zerovector, zerovector! as vi_zerovector!
import TimerOutputs: time as timer_time, ncalls as timer_ncalls, get_defaulttimer
import QuantumLattices: expand
import MPSKit: FiniteMPO, dot, correlator, transfer_left, transfer_right, AC_hamiltonian, AC2_hamiltonian, DerivativeOperator, local_update!
import MPSKit: left_virtualspace, right_virtualspace, physicalspace
import MPSKit: changebond, changebond!, changebonds, changebonds!
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
include("states/momentummps.jl")

include("utility/tools.jl")

include("algorithms/vectorops_threaded.jl")

include("finiteengine/config.jl")
include("finiteengine/environments.jl")
include("finiteengine/factorizations.jl")
include("finiteengine/localupdate.jl")
include("finiteengine/timeevolution.jl")
include("finiteengine/approximate.jl")

include("algorithms/dmrg.jl")
include("algorithms/hamiltonian_threaded.jl")
include("algorithms/cbe_expand.jl")
include("algorithms/excited.jl")
include("algorithms/cpt.jl")

include("utility/defaults.jl")

include("observables/correlator.jl")
include("observables/dcorrelator.jl")
include("observables/conductivity.jl")
include("observables/fourier.jl")
include("observables/linearprediction.jl")
include("observables/lehmann.jl")
include("observables/timesplit.jl")
include("observables/timesplit_distributed.jl")

# ── exports ──
export CustomLattice, BilayerSquare, Square, Custom, twosite_bonds, onesite_bonds, find_position, snake_2D, regroup_by_basis, kitaev_bonds
export hubbard, extended_hubbard, hubbard_bilayer_2band, kitaev_hubbard, heisenberg_model, JKGGp_model

export fZ, e_plus, e_min, hopping, cdagc, ccdag, σz_hopping, number, onsiteCoulomb, S_plus, S_min, S_z, S_square, neiborCoulomb, heisenberg, spinflip, pairhopping
export singlet_dagger, singlet, triplet_dagger, triplet
export chargedMPO, identityMPO, hamiltonian

export FiniteNormalMPS, FiniteSuperMPS, chargedMPS, chargedMPS!, identityMPS, randFiniteMPS, randInfiniteMPS
export MomentumMPS, check_momentum_mps, translation_phase, qmps

export add_single_util_leg, cart2polar, phase_by_polar, sort_by_distance, transfer_left, contract_MPO
export myDMRG1, myDMRG2, myTDVP1, myTDVP1_CBE, myTDVP2, myDMRG1_CBE
export CBEExpand
export dmrg1, dmrg1!, dmrg2, dmrg2!
export dmrg_mix, dmrg_mix!
export excited_dmrg1!, excited_dmrg2!, excited_states
export set_threaded_hamiltonian!, threaded_hamiltonian_threads
export HalfFiniteEnvironments, free_left!, free_right!, env_memory_bytes, configure_finite_engine!
export fast_timestep!, fast_approximate!
export Perioder, CPT, singleParticleGreenFunction, spectrum, densityofstates, GrandPotential, OrderParameters

export AbstractCorrelation, PairCorrelation, pair_amplitude_indices, TwoSiteCorrelation, OneSiteCorrelation, site_indices, correlator
export evolve_mps, dcorrelator, prepare_dcorrelator, sweep_dot
export conductivity
export fourier_kw, fourier_rw, fourier_rz, fourier_riw, static_structure_factor
export LinearPrediction, LinearPredictionModel, fit_linear_prediction, extend_series, extrapolate_series, validation_error, pole_analysis
export recommend_tmax, lehmann_eval, lehmann_report


function __init__()
    BLAS.set_num_threads(1)
    # Julia >= 1.12 forbids overwriting foreign same-signature methods during
    # precompilation, so the two TensorMap overrides (`inner`, `norm`) are
    # installed at load time instead (see algorithms/vectorops_threaded.jl)
    Core.eval(@__MODULE__, _THREADED_VECTOROPS_INIT)
end

end #module
