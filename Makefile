MAIN = main.cu
VEC3 = Vec3f.cu

PATH_SRC_VEC3 = srcVec3f
PATH_BASE = src1Base
PATH_STATIC = src2Static
PATH_FLOAT = src3Float
PATH_PINNED = src4Pinned
PATH_TRANSFER = src5Transfer
PATH_DIVERGENCE = src6Divergence
PATH_VALUE = src7Value
PATH_SHARED = src8Shared

OUT_BASE = gpu1Base.out
OUT_STATIC = gpu2Static.out
OUT_FLOAT = gpu3Float.out
OUT_PINNED = gpu4Pinned.out
OUT_TRANSFER = gpu5Transfer.out
OUT_DIVERGENCE = gpu6Divergence.out
OUT_VALUE = gpu7Value.out
OUT_SHARED = gpu8Shared.out

all: base static float pinned transfer divergence value shared

base:
	nvcc --default-stream per-thread -arch=sm_61 $(PATH_BASE)/$(MAIN) -o $(OUT_BASE)

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