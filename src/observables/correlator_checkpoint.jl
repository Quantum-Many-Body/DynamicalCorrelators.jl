# Raw-value checkpointing for the lazy CorrelationPlan driver.
#
# Same philosophy as the DMRG update-level checkpoint (algorithms/dmrg_checkpoint.jl):
# internal scratch in Serialization format, written atomically (tmp + rename),
# validated on load, and best-effort — a missing, corrupt, or mismatching file is
# ignored and the run simply computes everything, never a correctness risk.
#
# What is saved: the three raw caches plus the work counters, with a validation
# header (the state fingerprint from `_state_fingerprint` and one fingerprint per
# registered operator). Cache keys carry operator ids, which follow the channel
# registration order, so a changed script would silently reuse wrong values without
# the operator check; because ids are append-only, a rerun that ADDS channels still
# validates (the old fingerprint list is a prefix of the new one).

const _CORREL_CKPT_VERSION = 1

struct _CorrelCkpt
    path::String
    every::Int          # raw evaluations between checkpoint writes, per stage
end

# (scalar type, total codomain/domain dimension, norm): cheap identity of a local
# operator, enough to detect a changed operator table or registration order
function _op_fingerprint(op)
    return (nameof(scalartype(op)), dim(codomain(op)), dim(domain(op)), norm(op))
end

function _correl_ckpt_write(ckpt::_CorrelCkpt, plan::CorrelationPlan)
    payload = (
        version = _CORREL_CKPT_VERSION,
        N = length(plan.gs),
        fingerprint = _state_fingerprint(plan.gs),
        opfingerprints = [_op_fingerprint(op) for op in plan.operators],
        cache1 = plan.cache1, cache2 = plan.cache2, cache4 = plan.cache4,
        counters = map(getindex, plan.counters),
    )
    return _write_atomic(ckpt.path, payload)
end

# Merge a valid checkpoint into the plan caches; returns true on success. Any
# problem (missing/corrupt file, different state, changed operator table) leaves
# the plan untouched and returns false — the driver then computes everything.
function _correl_ckpt_load(ckpt::_CorrelCkpt, plan::CorrelationPlan, verbose::Integer)
    isfile(ckpt.path) || return false
    payload = try
        deserialize(ckpt.path)
    catch
        nothing
    end
    ok = payload isa NamedTuple &&
        get(payload, :version, 0) == _CORREL_CKPT_VERSION &&
        get(payload, :N, 0) == length(plan.gs) &&
        haskey(payload, :fingerprint) &&
        _fingerprint_match(payload.fingerprint, plan.gs) &&
        haskey(payload, :opfingerprints) &&
        length(payload.opfingerprints) <= length(plan.operators) &&
        all(eachindex(payload.opfingerprints)) do k
            payload.opfingerprints[k] == _op_fingerprint(plan.operators[k])
        end &&
        all(k -> haskey(payload, k), (:cache1, :cache2, :cache4, :counters))
    if !ok
        verbose > 0 && println(
            "  correlator checkpoint ignored (corrupt or from a different state/script): ",
            ckpt.path
        )
        return false
    end
    merge!(plan.cache1, payload.cache1)
    merge!(plan.cache2, payload.cache2)
    merge!(plan.cache4, payload.cache4)
    plan.counters.onesite[] += payload.counters.onesite
    plan.counters.twosite[] += payload.counters.twosite
    plan.counters.pair[] += payload.counters.pair
    verbose > 0 && @printf(
        "  correlator checkpoint: reused %d one-site, %d two-site, %d pair raw values from %s\n",
        length(payload.cache1), length(payload.cache2), length(payload.cache4), ckpt.path
    )
    return true
end
