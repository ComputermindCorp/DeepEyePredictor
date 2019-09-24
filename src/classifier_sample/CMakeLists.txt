cmake_minimum_required(VERSION 2.8)

set(TARGET classifier_sample)

project(${TARGET})

set(PREDICTOR_LIB_NAME deepeye_predictor_gpu)

if(WIN32)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" )
    set(CMAKE_SUPPRESS_REGENERATION true)
endif(WIN32)


##### cuda
find_package(CUDA REQUIRED)

if(UNIX)
    set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
    set(CUDA_LIBRARIE_DIRS "/usr/local/cuda/lib64")
    set(CUDA_LIBRARIES "-lcudart -lcublas -lcudnn")
elseif(WIN32)
    set(CUDA_ROOT $ENV{CUDA_PATH})
    set(CUDA_INCLUDE_DIRS ${CUDA_ROOT}/include)
    set(CUDA_LIBRARIE_DIRS ${CUDA_ROOT}/lib/x64)
    set(CUDA_LIBRARIES
        cudart.lib
        cublas.lib
        cudnn.lib
    )
endif(UNIX)


##### OPENCV
# OpenCV
if(UNIX)
    set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv/")
    set(OpenCV_LIBRARY_DIRS "/usr/local/lib/")
    set(OpenCV_LIBRARIES "-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs")        
elseif(WIN32)
    set(OpenCV_INCLUDE_DIRS "C:/opencv/build/include")
    set(OpenCV_LIBRARY_DIRS "C:/opencv/build/x64/vc14/lib")
    set(OpenCV_LIBRARIES
        "opencv_world343.lib"
        "opencv_world343d.lib"
    )
endif(UNIX)

##### INCLUDE
set(INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include" 
)
include_directories("${INCLUDE_DIRS}")

##### LIBRARY
if(UNIX)
    set(LIBRARY_DIRS
        ${CUDA_LIBRARIE_DIRS}
        ${OpenCV_LIBRARY_DIRS}
        "${CMAKE_CURRENT_SOURCE_DIR}/../../lib"
    )
elseif(WIN32)
    set(LIBRARY_DIRS
        ${CUDA_LIBRARIE_DIRS}
        ${OpenCV_LIBRARY_DIRS}
        "${CMAKE_CURRENT_SOURCE_DIR}/../../lib"
    )
endif(UNIX)

link_directories("${LIBRARY_DIRS}")

if(UNIX)
    set(LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        -ldl
        -lz
    )
elseif(WIN32)
    set(LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
    )
endif(UNIX)

##### FLAG
set(CMAKE_CXX_FLAGS "-std=c++11")
if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -NODEFAULTLIB:MSVCRTD")
endif(UNIX)
set(CMAKE_CXX_FLAGS_DEBUG, "-g")
set(CMAKE_CXX_FLAGS_RELEASE, "-O2 -march=native")

#
add_executable(${TARGET} classifier_sample.cpp cm_image.cpp )

target_link_libraries(${TARGET} ${PREDICTOR_LIB_NAME} ${LIBS})
