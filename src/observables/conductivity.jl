#conductivity based on single-particle Green's function: A. Georges, G. Kotliar, W. Krauth, and M. J. Rozenberg, Dynamical Mean-Field Theory of Strongly Correlated Fermion Systems and the Limit of Infinite Dimensions, Rev. Mod. Phys. 68, 13 (1996).

function conductivity(∂Ek::AbstractArray, gf::AbstractArray, ω_range::AbstractArray)
    f(ω) = 1/(exp(β*ω)+1)
    ∂f(ω) = -β*f(ω)*(1-f(ω))
    σ = 0
    for k in eachindex(∂Ek)
        for (i, ω) in enumerate(ω_range)
            σ += ((∂Ek[k])^2)*((gf[k][i])^2)*(-∂f(ω))
        end
    end
    return σ
end





#conductivity based on current-current correlations