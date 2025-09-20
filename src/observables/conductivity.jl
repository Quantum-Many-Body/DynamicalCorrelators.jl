#conductivity based on single-particle Green's function: A. Georges, G. Kotliar, W. Krauth, and M. J. Rozenberg, Dynamical Mean-Field Theory of Strongly Correlated Fermion Systems and the Limit of Infinite Dimensions, Rev. Mod. Phys. 68, 13 (1996).

function conductivity(β::Number, ∂Ek::AbstractArray, Akw::AbstractArray, ω_range::AbstractArray; μ=0, ifsum=false)
    f(ω) = 1/(exp(β*(ω-μ))+1)
    ∂f(ω) = -β*f(ω)*(1-f(ω))
    pf = [-∂f(ω) for ω in ω_range]
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





#conductivity based on current-current correlations