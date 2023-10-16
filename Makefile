MAIN = main.cu

PATH_SRC_VEC3 = srcVec3f
PATH_BASE = src1Base
PATH_THREADS = src2Threads
PATH_STATIC = src3Static
PATH_FLOAT = src4Float
PATH_PINNED = src5Pinned
PATH_TRANSFER = src6Transfer
PATH_DIVERGENCE = src7Divergence
PATH_VALUE = src8Value
PATH_SHARED = src9Shared

OUT_BASE = gpu1Base.out
OUT_THREADS = gpu2Threads.out
OUT_STATIC = gpu3Static.out
OUT_FLOAT = gpu4Float.out
OUT_PINNED = gpu45inned.out
OUT_TRANSFER = gpu6Transfer.out
OUT_DIVERGENCE = gpu7Divergence.out
OUT_VALUE = gpu8Value.out
OUT_SHARED = gpu9Shared.out

all: base threads static float pinned transfer divergence value shared

base:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_BASE)/$(MAIN) -o $(OUT_BASE)

threads:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_THREADS)/$(MAIN) -o $(OUT_THREADS)

static:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_STATIC)/$(MAIN) -o $(OUT_STATIC)

float:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_FLOAT)/$(MAIN) -o $(OUT_FLOAT)

pinned:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_PINNED)/$(MAIN) -o $(OUT_PINNED)

transfer:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_TRANSFER)/$(MAIN) -o $(OUT_TRANSFER)

divergence:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_DIVERGENCE)/$(MAIN) -o $(OUT_DIVERGENCE)

value:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_VALUE)/$(MAIN) -o $(OUT_VALUE)

shared:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_SHARED)/$(MAIN) -o $(OUT_SHARED)

clean:
	rm -f $(OUT_BASE) $(OUT_STATIC) $(OUT_FLOAT) $(OUT_PINNED) $(OUT_TRANSFER) $(OUT_DIVERGENCE) $(OUT_VALUE) $(OUT_SHARED) img.ppm