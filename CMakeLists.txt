cmake_minimum_required(VERSION 3.10)
project(CutlassTest LANGUAGES CXX)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(BUILD_EXECUTABLE name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE /usr/local/cuda/include)
    target_include_directories(${name} PRIVATE /home/xiaoyu/workspace/cutlass/include)
    target_include_directories(${name} PRIVATE /home/xiaoyu/workspace/cutlass/tools/util/include)
    target_link_libraries(${name} cudart)

    add_custom_command(OUTPUT ${name}.ptx
        COMMAND /usr/local/cuda/bin/cuobjdump  -ptx ${CMAKE_CURRENT_BINARY_DIR}/${name} > ${CMAKE_CURRENT_BINARY_DIR}/${name}.ptx
        COMMAND /usr/local/cuda/bin/cuobjdump  -sass ${CMAKE_CURRENT_BINARY_DIR}/${name} > ${CMAKE_CURRENT_BINARY_DIR}/${name}.sass
    )
    add_custom_target(OUTPUT_ASM_${name} ALL DEPENDS ${name}.ptx)
    add_dependencies(OUTPUT_ASM_${name} ${name})
    # message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

# BUILD_EXECUTABLE(00_sgemm ${PROJECT_SOURCE_DIR}/src/00_sgemm.cu)

BUILD_EXECUTABLE(turing_tensorop_gemm ${PROJECT_SOURCE_DIR}/src/turing_tensorop_gemm.cu)
# BUILD_EXECUTABLE(volta_tensorop_gemm ${PROJECT_SOURCE_DIR}/src/volta_tensorop_gemm.cu)
# BUILD_EXECUTABLE(ampere_tf32_tensorop_gemm ${PROJECT_SOURCE_DIR}/src/ampere_tf32_tensorop_gemm.cu)

BUILD_EXECUTABLE(gemm_fp16_sm80 ${PROJECT_SOURCE_DIR}/src/gemm_fp16_sm80.cu)
BUILD_EXECUTABLE(gemm_int8_sm80 ${PROJECT_SOURCE_DIR}/src/gemm_int8_sm80.cu)