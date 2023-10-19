# CUDAOptimizedRayTracer
Project to optimize the CUDA ray tracer found in https://github.com/xkevio/CUDA-Raytracer.

Iterations of optimizations has been split up into different directories, different code versions, in numerical order to allow for simple comparisons, these being src followed by a number and a name. Each directory includes a README specifying changes. Meanwhile, srcVec3f and srcFloat3 are used by multiple iterations.

Use

```
make
```

to compile all versions. Then run

```
run.sh
```

to easily be able to select any of the optimization version as well as one of the different profiling tools used in the project. The different versions have the following dependencies where the path leading from src1Base to src7Shared is considered the final path of the project:

<pre>
src1Base <br />
    | <br />
src2Static <br />
    | <br />
src3Threads <br />
    | <br />
src4Float <br />
    | <br />
src5Pinned <br />
    | <br />
src6Transfer ----- src7Shared <br />
    | <br />
src7Divergence <br />
    | <br />
src8Value <br />
    | <br />
src9Shared
</pre>
