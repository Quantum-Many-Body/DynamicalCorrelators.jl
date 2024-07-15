# DCorrelators.jl

*A convenient frontend for calculating dynamical correlation functions and related observables based on matrix-product states time evolution methods.*

## Introduction

- The symbolic operator representation of a quantum lattice system in condensed matter physics is based on the package [`QuantumLattices`](https://github.com/Quantum-Many-Body/QuantumLattices.jl.git)

- The matrix-product states time evolution methods such as TEBD, MPO $W^{II}$ and TDVP are based on packages   [`ITensors`](https://github.com/ITensor/ITensors.jl.git) and [`MPSKit`](https://github.com/QuantumKitHub/MPSKit.jl.git)

- The bechmark of dynamical correlation functions and related observables is the result from exact diagonalization method based on the packages [`ExactDiagonalization`](https://github.com/Quantum-Many-Body/ExactDiagonalization.jl.git)



## Installation

Please type `]` in the REPL to use the package mode, then type this command:

```julia
pkg> add DCorrelators
```
*Currently unavailable*

## Discrete space an time Fourier transforms

If the $x$ variable has only discrete values ($x=na$,for $n=1,2,3,...,N$) and finite length $L$ ($L=Na$), the expansion of the function is
$$f_n=\sum_{m=1}^{N} A_{q} e^{iq_m x_n},\quad q=\frac{2\pi}{L}m,$$
where 
$$A_{q}=\frac{1}{N}\sum^N_{n=1}f_n e^{-iq_mx_n}. $$
Dividing $A_{q}$ by the mode $\frac{2\pi}{L}$, the Fourier amplitudes with a per unit spacial interval is
$$ A(q) = \frac{L}{2\pi}A_q=\frac{a}{2\pi}\sum^N_{n=1}f_ne^{-iqx_n}$$

If the times $t$ are discrete times ($t=l\Delta t$, for $l=0,1,2,...,N$) and the final evolutionary time $t_{\mathrm{end}}=N\Delta t$, the expansion of the function is
$$ f_l=\sum_{p=1}^N A_{\omega} e^{-i\omega_p t_l},\quad \omega=\frac{2\pi}{t_{\mathrm{end}}}p,$$
where
$$A_{\omega}=\frac{1}{N}\sum_{l=1}^N f_l e^{i\omega_p t_l}.$$
To make it a per unit frequency interval, one need to divide by the spacing of the discrete frequency mode and the Fourier amlitudes are given by,
$$A(\omega)=\frac{t_{\mathrm{end}}}{2\pi}A_{\omega}=\frac{\Delta t}{2\pi}\sum_{l=1}^N f_l e^{i\omega t_l}.$$
Although a Fourier series is designed to represent functions that are periodic, one can assume that the finite data sequence can be periodically repeated, which leads to the time at index $l=N$ is identified with the time at $l=0$. However, the small errors made at the end of a period will be irrelevant as long as the primary correlations decay in less time than $t_{\mathrm{end}}$. 

## Dynamical correlation functions


## Tutorial

### Quantum lattice
*come soon*

### Hamiltonian
*come soon*

### Correlations
*come soon*

## Note

Due to the fast development of this package, releases with different minor version numbers are **not** guaranteed to be compatible with previous ones **before** the release of v1.0.0. Comments are welcomed in the issues.

## Contact

zongyy_phy@smail.nju.edu.cn