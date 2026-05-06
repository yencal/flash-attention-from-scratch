NVCC = nvcc
ARCH = sm_80
CFLAGS = -O3 -arch=$(ARCH) --std=c++17 --use_fast_math
TARGET = flash_attention

$(TARGET): src/main.cu src/*.cuh
	$(NVCC) $(CFLAGS) -lcurand -o $@ src/main.cu

profile: src/profile.cu src/*.cuh
	$(NVCC) $(CFLAGS) -lcurand -o $@ src/profile.cu

autotune: src/autotune.cu src/*.cuh
	$(NVCC) $(CFLAGS) -lcurand -o $@ src/autotune.cu

clean:
	rm -f $(TARGET) profile autotune *.csv

.PHONY: clean
