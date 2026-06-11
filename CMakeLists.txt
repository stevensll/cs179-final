cmake_minimum_required(VERSION 3.24)
project(cs179_final LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

find_package(Threads REQUIRED)

add_executable(cpu_demo
    src/cpu/cpu_demo.cpp
    src/cpu/cpu_pipeline.cpp
    src/common/audio.cpp
)
target_include_directories(cpu_demo PRIVATE ${SNDFILE_INCLUDE_DIRS})
target_compile_options(cpu_demo PRIVATE -O3 -Wall -Wextra)
target_link_libraries(cpu_demo PRIVATE ${SNDFILE_LIBRARIES} Threads::Threads)

add_executable(gpu_detect
    src/gpu/gpu_detect.cu
    src/gpu/gpu_pipeline.cu
    src/gpu/kernels.cu
    src/gpu/gemm.cu
    src/common/audio.cpp
)
target_include_directories(gpu_detect PRIVATE ${SNDFILE_INCLUDE_DIRS})
target_compile_options(gpu_detect PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info>
)
target_link_libraries(gpu_detect PRIVATE CUDA::cufft CUDA::cublas ${SNDFILE_LIBRARIES} Threads::Threads)
