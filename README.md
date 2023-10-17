# CUDAOptimizedRayTracer
Project to optimize the CUDA ray tracer found in https://github.com/xkevio/CUDA-Raytracer.

Iterations of optimizations has been split up into different directories, different code versions, in numerical order to allow for simple comparisons, these being src followed by a number and a name. Each directory includes a README specifying changes. Meanwhile, srcVec3f and srcFloat3 are used by multiple iterations.

The versions have the following dependencies:

src1Base  
    |  
src2Static  
    |  
src3Threads  
    |  
src4Float  
    |  
src5Pinned  
    |  
src6Transfer ----- src7Shared  
    |  
src7Divergence  
    |  
src8Value  
    |  
src9Shared
