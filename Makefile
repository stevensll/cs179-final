# Thin wrapper so graders can just type `make` (build system is CMake).

all:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j

clean:
	rm -rf build

.PHONY: all clean
