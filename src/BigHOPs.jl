module BigHOPs

export Hyperoptimizer, run!, settarget!
export Stateful
export RandomSampler, LHSampler
export Hyperband, ASHA, DEHB
export Serial, Threaded, DistributedQueue
export minimizer, history, results, printmin
export summaryplot
export Nominal, Ordinal, Continuous
export save_hyperoptimizer, load_hyperoptimizer

using Random
using Printf
using Distributed
using StableRNGs: StableRNG
import JLD2
import ProgressMeter
import DataFrames
import QuasiMonteCarlo

include("basic/domains.jl")
include("basic/types.jl")

include("samplers/sampler.jl")
include("samplers/random.jl")
include("samplers/lhs.jl")

"""
    BasicSamplers

Samplers that draw each configuration directly: `RandomSampler` and `LHSampler`. These are the `inner` samplers `Hyperband` and `ASHA` accept.
"""
const BasicSamplers = Union{LHSampler,RandomSampler}

include("samplers/sha_based/sh.jl")
include("samplers/sha_based/synchronous.jl")
include("samplers/sha_based/asynchronous.jl")
include("samplers/sha_based/dehb_sync.jl")
"""
    FixedPlanSampler

Samplers whose trial count is fixed when the `Hyperoptimizer` is built (`LHSampler` and the successive-halving samplers), so `settarget!` rejects them.
"""
const FixedPlanSampler = Union{LHSampler,SuccessiveHalving}

include("executors/executor.jl")
include("executors/serial.jl")
include("executors/threaded.jl")
include("executors/distributed_queue.jl")

include("basic/optimizer.jl")

include("basic/report.jl")
include("basic/persistence.jl")
include("basic/dataframe.jl")
include("basic/plotting.jl")

end
