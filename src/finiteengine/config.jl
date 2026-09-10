# Parallel configuration and memory introspection for the fast finite engine.
#
# Recommended HPC setup (the FiniteMPS.jl philosophy, and what benchmarks on
# this project showed to be faster): many Julia threads for the coarse-grained
# parallelism (Hamiltonian/environment channels, SVD blocks), BLAS pinned to a
# single thread per task. `configure_finite_engine!` makes this explicit.
# TensorKit's SU(2) recoupling ("transformer") and index-manipulation threads
# are a separate, orthogonal thread pool that can be enabled on top.

"""
    configure_finite_engine!(; blas_threads = 1, transformer_threads = nothing,
        manipulation_threads = nothing, verbose = true)

Set the thread configuration recommended for the fast finite engine:

- `blas_threads`: BLAS/LAPACK threads per call (default `1`; the engine
  parallelizes above BLAS, so multi-threaded BLAS only adds contention)
- `transformer_threads`: TensorKit recoupling (fusion-tree transformation)
  threads, or `nothing` to leave unchanged
- `manipulation_threads`: TensorKit index-manipulation threads, or `nothing`
  to leave unchanged
- `verbose`: print the resulting configuration

Returns the number of Julia threads (the pool the engine actually uses).
"""
function configure_finite_engine!(;
        blas_threads::Integer = 1,
        transformer_threads = nothing,
        manipulation_threads = nothing,
        verbose::Bool = true
    )
    BLAS.set_num_threads(Int(blas_threads))
    if transformer_threads !== nothing
        set_num_transformer_threads(Int(transformer_threads))
    end
    if manipulation_threads !== nothing
        set_num_manipulation_threads(Int(manipulation_threads))
    end
    if verbose
        println("Finite-engine parallel configuration:")
        println("  Julia threads:               ", Threads.nthreads())
        println("  BLAS threads:                ", BLAS.get_num_threads())
        println("  TensorKit transformer threads:  ", get_num_transformer_threads())
        println("  TensorKit manipulation threads: ", get_num_manipulation_threads())
        flush(stdout)
    end
    return Threads.nthreads()
end

# ---------------------------------------------------------------------------
# memory introspection
# ---------------------------------------------------------------------------

"""
Current process resident set size in bytes. Linux reads `VmRSS` from
`/proc/self/status` (true current RSS); other platforms fall back to
`Sys.maxrss()`, which is the *peak* RSS — it never decreases, so treat it as an
upper bound there.
"""
function _rss_bytes()
    if Sys.islinux()
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                return parse(Int, split(line)[2]) * 1024
            end
        end
    end
    return Sys.maxrss()
end

_format_gib(bytes::Real) = @sprintf("%.2f GiB", bytes / 2^30)

# Print a tagged memory line: live environment bytes plus process RSS.
function _print_mem(tag::AbstractString, env = nothing)
    envpart = env === nothing ? "" :
        " | env stored ≈ $(_format_gib(env_memory_bytes(env)))"
    println("  [mem] ", tag, envpart, " | RSS ≈ $(_format_gib(_rss_bytes()))")
    flush(stdout)
    return nothing
end
