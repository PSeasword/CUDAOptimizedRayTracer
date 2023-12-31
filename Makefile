MAIN = main.cu

PATH_BASE = src1Base
PATH_STATIC = src2Static
PATH_THREADS = src3Threads
PATH_FLOAT = src4Float
PATH_PINNED = src5Pinned
PATH_TRANSFER = src6Transfer
PATH_SHARED_OPTIONAL = src7Shared
PATH_DIVERGENCE = src7Divergence
PATH_VALUE = src8Value
PATH_SHARED = src9Shared

OUT_BASE = gpu1Base
OUT_STATIC = gpu2Static
OUT_THREADS = gpu3Threads
OUT_FLOAT = gpu4Float
OUT_PINNED = gpu5Pinned
OUT_TRANSFER = gpu6Transfer
OUT_SHARED_OPTIONAL = gpu7Shared
OUT_DIVERGENCE = gpu7Divergence
OUT_VALUE = gpu8Value
OUT_SHARED = gpu9Shared

all: base static threads float pinned transfer shared_optional divergence value shared

base:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_BASE)/$(MAIN) -o $(OUT_BASE).out

static:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_STATIC)/$(MAIN) -o $(OUT_STATIC).out

threads:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_THREADS)/$(MAIN) -o $(OUT_THREADS).out

float:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_FLOAT)/$(MAIN) -o $(OUT_FLOAT).out

pinned:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_PINNED)/$(MAIN) -o $(OUT_PINNED).out

transfer:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_TRANSFER)/$(MAIN) -o $(OUT_TRANSFER).out

shared_optional:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_SHARED_OPTIONAL)/$(MAIN) -o $(OUT_SHARED_OPTIONAL).out

divergence:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_DIVERGENCE)/$(MAIN) -o $(OUT_DIVERGENCE).out

value:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_VALUE)/$(MAIN) -o $(OUT_VALUE).out

shared:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_SHARED)/$(MAIN) -o $(OUT_SHARED).out

clean:
	rm -f $(OUT_BASE).out $(OUT_STATIC).out $(OUT_THREADS).out $(OUT_FLOAT).out $(OUT_PINNED).out $(OUT_TRANSFER).out $(OUT_SHARED_OPTIONAL).out $(OUT_DIVERGENCE).out $(OUT_VALUE).out $(OUT_SHARED).out
	rm -f $(OUT_BASE).nvvp $(OUT_STATIC).nvvp $(OUT_THREADS).nvvp $(OUT_FLOAT).nvvp $(OUT_PINNED).nvvvp $(OUT_TRANSFER).nvvp $(OUT_SHARED_OPTIONAL).nvvp $(OUT_DIVERGENCE).nvvp $(OUT_VALUE).nvvp $(OUT_SHARED).nvvp
	rm -f tmp.txt img.ppm
