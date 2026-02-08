"""
    conductivity(β, ∂Ek, Akw, ω_range; μ=0, ifsum=false)

Compute the DC optical conductivity from the single-particle spectral function using the
Kubo formula in the DMFT approximation.

Reference: A. Georges, G. Kotliar, W. Krauth, and M. J. Rozenberg,
*Dynamical Mean-Field Theory of Strongly Correlated Fermion Systems and the Limit of
Infinite Dimensions*, Rev. Mod. Phys. 68, 13 (1996).

``\\sigma = \\sum_k (\\partial E_k)^2 \\int d\\omega \\, (-\\partial f(\\omega)) \\, A^2(k, \\omega)``

# Arguments
- `β`: inverse temperature `1/T`.
- `∂Ek`: array of band velocity (energy derivative) for each k-point.
- `Akw`: array of spectral functions `A(k, ω)` for each k-point (each element is a vector over ω).
- `ω_range`: frequency grid.
- `μ`: chemical potential (default: 0).
- `ifsum`: if `true`, use simple summation; if `false`, use numerical integration.
"""
function conductivity(β::Number, ∂Ek::AbstractArray, Akw::AbstractArray, ω_range::AbstractArray; μ=0, ifsum=false)
    f(ω) = 1/(exp(β*(ω-μ))+1)          # Fermi-Dirac distribution
    ∂f(ω) = -β*f(ω)*(1-f(ω))           # derivative of Fermi-Dirac
    pf = [-∂f(ω) for ω in ω_range]     # -∂f/∂ω evaluated on the grid
    σ = 0
    if ifsum
        for k in eachindex(∂Ek)
            temp = pf .* ((Akw[k]).^2)
            σ += ((∂Ek[k])^2)*sum(temp)
        end
    else
        for k in eachindex(∂Ek)
            temp = pf .* ((Akw[k]).^2)
            σ += ((∂Ek[k])^2)*integrate(ω_range, temp)
        end
    end
    return σ
end




