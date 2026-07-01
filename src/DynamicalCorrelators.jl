module DynamicalCorrelators

using LinearAlgebra: norm, inv, mul!, I, tr, dot, BLAS, logabsdet
using QuantumLattices: Hilbert, Term, Lattice, Neighbors, azimuth, rcoordinate, bonds, Bond, OperatorGenerator, Operator, CompositeIndex, CoordinatedIndex, FockIndex, Index, OperatorSet
using QuantumLattices: AbstractLattice as QLattice, Table, isintracell, OperatorIndexToTuple, icoordinate, ReciprocalSpace, issubordinate
using TensorOperations: promote_contract, tensorfree!
using TensorKit: FermionParity, Trivial, U1Irrep, SU2Irrep, SU2Space, Vect, Sector, ProductSector, AbstractTensorMap, TensorMap, BraidingStyle, BraidingTensor, sectortype, sectors, Bosonic
using TensorKit: truncrank, truncerror, trunctol, ←, space, numout, numin, dual, fuse, svd_trunc!, svd_compact!, normalize!,normalize, oneunit, notrunc, similarstoragetype, insertleftunit, insertrightunit, removeunit
using TensorKit: left_null, right_null!, catdomain, catcodomain, qr_compact!, left_orth, right_orth, rmul!
using TensorKit: ⊠, ⊗, permute, repartition, domain, codomain, isomorphism, isometry, storagetype, @plansor, @planar, @tensor, blocks, block, flip, dim, infimum, id, zerovector, tensormaptype
using BlockTensorKit: nonzero_pairs, nonzero_length
using MPSKit: FiniteMPS, InfiniteMPS, FiniteMPOHamiltonian, MPOHamiltonian, TDVP, TDVP2, DMRG, DMRG2, IDMRG, IDMRG2, changebonds!, SvdCut, left_virtualspace, right_virtualspace
using MPSKit: Algorithm, add_util_leg, _firstspace, _lastspace, decompose_localmpo, TransferMatrix, environments, expectation_value, max_virtualspaces, physicalspace
using MPSKit: spacetype, fuse_mul_mpo, fuser, DenseMPO, MPOTensor, approximate, LAPACK_DivideAndConquer, left_orth!, right_orth!
using MPSKit.Defaults: _finalize
using MPSKit: AbstractFiniteMPS, updatetol, zerovector!, _transpose_front, MPSTensor, MPSBondTensor, check_unambiguous_braiding, scalartype, fixedpoint, transfer_leftenv!, transfer_rightenv!, transfer_right
using MPSKit: _mul_tail, _mul_front, _transpose_tail, AC2, recalculate!, calc_galerkin, IDMRGState, IterativeSolver
using MPSKit: leftenv, rightenv, JordanMPO_AC_Hamiltonian, JordanMPOTensor
using KrylovKit: exponentiate, eigsolve, Lanczos, ModifiedGramSchmidt
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
import MPSKit: FiniteMPO, propagator, dot, correlator, transfer_left, transfer_right, AC_hamiltonian, AC2_hamiltonian, C_hamiltonian, DerivativeOperator, timestep, timestep!, integrate as mps_integrate
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

include("algorithms/dmrg2.jl")
include("algorithms/dmrg1_cbe.jl")
include("algorithms/tdvp1_cbe.jl")
include("algorithms/idmrg2.jl")
include("algorithms/cpt.jl")
include("algorithms/hamiltonian_derivatives_multithreading.jl")

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
export myDMRG, myDMRG2, myTDVP, myTDVP1_CBE, myTDVP2, myDMRG1CBE_eigsolve

export dmrg2!, dmrg2, dmrg2_sweep!
export dmrg1_cbe!, dmrg1_cbe
export TDVP1_CBE
export idmrg2
export Perioder, CPT, singleParticleGreenFunction, spectrum, densityofstates, GrandPotential

export AbstractCorrelation, PairCorrelation, pair_amplitude_indices, TwoSiteCorrelation, OneSiteCorrelation, site_indices, correlator
export evolve_mps, dcorrelator, sweep_dot
export conductivity
export fourier_kw, fourier_rw, fourier_rz, fourier_riw, static_structure_factor

function __init__()
    BLAS.set_num_threads(1)
end

end #module
