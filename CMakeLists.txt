cmake_minimum_required(VERSION 3.2)
SET(CMAKE_CXX_STANDARD 14)
PROJECT(HypER)
set(CMAKE_C_FLAGS "-std=c99")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g")

find_package(Torch
    PATHS lib/REENet/lib/libtorch
        )
macro(find_libraries)
    # find MPI----------------------------------------------------------------------
    find_package(MPI REQUIRED)
    if("${MPI_CXX_INCLUDE_PATH}" MATCHES "open")
        message(STATUS "Using OpenMPI.")
        add_definitions(-DOPEN_MPI)
    endif()
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

    # find OpenMP----------------------------------------------------------------------
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()

    # find Threads------------------------------------------------------------------
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    find_package(Threads REQUIRED)

    # find glog---------------------------------------------------------------------
    include("cmake/FindGlog.cmake")

    # find gflags-------------------------------------------------------------------
    #include("cmake/FindGFlags.cmake")

    # find tcmalloc-----------------------------------------------------------------
    include("cmake/FindGperftools.cmake")
    if (NOT GPERFTOOLS_FOUND)
        add_definitions(-DNO_TCMALLOC)
    endif ()

    # find torch-----------------------------------------------------------------
    # find boost--------------------------------------------------------------------
    # find_package(Boost COMPONENTS system filesystem)
    # include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

    # find gtest--------------------------------------------------------------------
    # include("cmake/FindGTest.cmake")

endmacro(find_libraries)

find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
	set(CUDA_SOURCE_PROPERTY_FORMAT OBJ)
	set(CUDA_SEPARABLE_COMPILATION ON)
	include_directories(${CUDA_INCLUDE_DIRS})
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS -arch=sm_61;-O2;-G;-g;-std=c++14)#根据具体GPU性能更改算力参数
	link_directories($ENV{CUDA_PATH}/lib/x64)
else(${CUDA_FOUND})
	MESSAGE(STATUS "cuda not found!")
endif(${CUDA_FOUND})

#LINK_DIRECTORIES(lib/REENet/lib/cfasttext/build/)


include_directories(CUDA_INCLUDE ${CUDA_PATH})

include_directories(lib/sqlite3/include)
include_directories(lib/REENet/lib/rapidcsv/src)
include_directories(lib/REENet/include)
include_directories(lib/REENet/lib/cfasttext/include)
include_directories(include)
include_directories(lib/REENet/lib/libtorch/include)
include_directories(lib/REENet/lib/libtorch/include/torch)
include_directories(lib/REENet/lib/libtorch/include/c10/core)
include_directories(lib/REENet/lib/libtorch/include/c10/util)
include_directories(lib/REENet/lib/libtorch/include/c10/cuda)
include_directories(lib/REENet/lib/libtorch/include/c10/macros)
include_directories(/home/LAB/zhuxk/tools/mpi/include)


aux_source_directory(lib/REENet/lib/cfasttext/build/fasttext FASTTEXT)
aux_source_directory(src/core CORE)
aux_source_directory(src/load_management LOADER)
aux_source_directory(src/matcher MATCHER)
aux_source_directory(src/scheduling SCHEDULING)
aux_source_directory(src/fixpoint_model FIXPOINT)
#aux_source_directory(src/tools TOOLS)
aux_source_directory(lib/REENet/src REENet)
#aux_source_directory(examples/ TEST)

#cuda_add_executable(cuer examples/test_songs.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER})
#cuda_add_executable(cuer  examples/cudaGetDeviceProperties.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER})
#cuda_add_executable(cuer  examples/eva_dblp_scholar.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER})
#cuda_add_executable(cuer  examples/eva_dblp_acm.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER})
#cuda_add_executable(cuer  examples/eva_walmart.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER})
#cuda_add_executable(cuer  examples/eva_amazon_google.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER})
#cuda_add_executable(cuer  examples/eva_songs.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${FIXPOINT})
#cuda_add_executable(cuer  examples/eva_ocp.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER})
#cuda_add_executable(cuer  examples/REE/eva_dblp_acm_ree.cu ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER})

#example_MPI_AND_CUDA
#cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_dblp_acm_threads.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})
#cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_walmart_threads.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})
#cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_imdb_omdb_threads2.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})
#cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_imdb_omdb.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})
cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_songs_threads.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})
#cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_ocp_threads.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})
#cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_tfacc_threads.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})
#cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_tpch_threads.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})
#cuda_add_executable(${PROJECT_NAME}  examples/hsa/eva_tpch_threads2.cc ${CUDA_INCLUDE} ${LOADER} ${CORE} ${MATCHER} ${SCHEDULING})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -pthread" )


target_link_libraries(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/lib/sqlite3/lib/libsqlite3.so ${PROJECT_SOURCE_DIR}/lib/REENet/build/libreenet.so "${TORCH_LIBRARIES}" /home/LAB/zhuxk/tools/mpi/lib/libmpi.so ${MPI_CXX_LIBRARIES})
