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

OUT_BASE = gpu1Base
OUT_THREADS = gpu2Threads
OUT_STATIC = gpu3Static
OUT_FLOAT = gpu4Float
OUT_PINNED = gpu5Pinned
OUT_TRANSFER = gpu6Transfer
OUT_DIVERGENCE = gpu7Divergence
OUT_VALUE = gpu8Value
OUT_SHARED = gpu9Shared

all: base threads static float pinned transfer divergence value shared

base:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_BASE)/$(MAIN) -o $(OUT_BASE).out

threads:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_THREADS)/$(MAIN) -o $(OUT_THREADS).out

static:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_STATIC)/$(MAIN) -o $(OUT_STATIC).out

float:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_FLOAT)/$(MAIN) -o $(OUT_FLOAT).out

pinned:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_PINNED)/$(MAIN) -o $(OUT_PINNED).out

transfer:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_TRANSFER)/$(MAIN) -o $(OUT_TRANSFER).out

divergence:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_DIVERGENCE)/$(MAIN) -o $(OUT_DIVERGENCE).out

value:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_VALUE)/$(MAIN) -o $(OUT_VALUE).out

shared:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_SHARED)/$(MAIN) -o $(OUT_SHARED).out

clean:
	rm -f $(OUT_BASE).out $(OUT_THREADS).out $(OUT_STATIC).out $(OUT_FLOAT).out $(OUT_PINNED).out $(OUT_TRANSFER).out $(OUT_DIVERGENCE).out $(OUT_VALUE).out $(OUT_SHARED).out
	rm -f $(OUT_BASE).nvvp $(OUT_THREADS).nvvp $(OUT_STATIC).nvvp $(OUT_FLOAT).nvvp $(OUT_PINNED).nvvvp $(OUT_TRANSFER).nvvp $(OUT_DIVERGENCE).nvvp $(OUT_VALUE).nvvp $(OUT_SHARED).nvvp
	rm -f tmp.txt img.ppm
