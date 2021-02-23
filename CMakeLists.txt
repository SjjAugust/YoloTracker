cmake_minimum_required(VERSION 3.5)

# ----------------------------------------------------------------------------
# Set Project Name
# ----------------------------------------------------------------------------
set (PROJECT_NAME  YoloTracker)
# ----------------------------------------------------------------------------
# Setup CUDA
# ----------------------------------------------------------------------------
find_package(CUDA)
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53  #tegra tx1 
        -gencode arch=compute_61,code=sm_61  #gtx 1060
	-gencode arch=compute_62,code=sm_62  #tegra tx2
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")
	set (
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS};
		-gencode arch=compute_72,code=sm_72 #xavier
                -gencode arch=compute_75,code=sm_75 #rtx2080 
	)
endif()
# ----------------------------------------------------------------------------
# Find Libraries
# ----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/data/work/YoloTracker/cmake/")
FIND_PACKAGE(Caffe REQUIRED)
FIND_PACKAGE(OpenCV 3.3.1 REQUIRED)
# ----------------------------------------------------------------------------
# Add Libraries and peoject files
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${Caffe_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/usr/include)
LINK_DIRECTORIES(/usr/lib)
LINK_LIBRARIES(glog gflags boost_filesystem boost_system)

INCLUDE(CheckIncludeFileCXX)
check_include_file_cxx(opencv2/bgsegm.hpp HAVE_OPENCV_CONTRIB)
message("-- HAVE_OPENCV_CONTRIB: ${HAVE_OPENCV_CONTRIB}")

INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/Tracker
        ${PROJECT_SOURCE_DIR}/Tracker/HungarianAlg
        ${PROJECT_SOURCE_DIR}/Tracker/graph
        ${PROJECT_SOURCE_DIR}/Tracker/graph/GTL/include
        ${PROJECT_SOURCE_DIR}/yolo
	/usr/local/cuda/include
)

# ----------------------------------------------------------------------------
# Add Definition and Options
# ----------------------------------------------------------------------------
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
option(USE_OPENCV "Build with OpenCV support" ON)
if(HAVE_OPENCV_CONTRIB)
    add_definitions(-DHAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" ON)
    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" ON)
    option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" ON)

else(HAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" OFF)
    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" OFF)
    option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" OFF)
endif(HAVE_OPENCV_CONTRIB)

if(USE_OCV_UKF)
    add_definitions(-DUSE_OCV_UKF)
endif(USE_OCV_UKF)

if(USE_OCV_KCF)
    add_definitions(-DUSE_OCV_KCF)
endif(USE_OCV_KCF)

if(USE_OCV_BGFG)
    add_definitions(-DUSE_OCV_BGFG)
endif(USE_OCV_BGFG)
# ----------------------------------------------------------------------------
# SELECT C++ standard
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)

# ----------------------------------------------------------------------------
# Link source and header files
# ----------------------------------------------------------------------------
set(graph_source
        Tracker/graph/tokenise.cpp
        #             Tracker/graph/script.cpp
        #             Tracker/graph/rings.cpp
        Tracker/graph/mytree.cpp
        Tracker/graph/mygraph.cpp
        Tracker/graph/mwbmatching.cpp
        Tracker/graph/mincut.cpp
        Tracker/graph/gport.cpp
        #             Tracker/graph/gml2nestedsql.cpp
        #             Tracker/graph/gml2dot.cpp
        #             Tracker/graph/components.cpp
        Tracker/graph/fheap.c
        )

set(graph_header
        Tracker/graph/tokenise.h
        Tracker/graph/mytree.h
        Tracker/graph/mygraph.h
        Tracker/graph/mwbmatching.h
        Tracker/graph/mincut.h
        Tracker/graph/gport.h
        Tracker/graph/gdefs.h
        Tracker/graph/fheap.h
        )

set(gtl_source
        Tracker/graph/GTL/src/bellman_ford.cpp
        Tracker/graph/GTL/src/bfs.cpp
        Tracker/graph/GTL/src/biconnectivity.cpp
        Tracker/graph/GTL/src/bid_dijkstra.cpp
        Tracker/graph/GTL/src/components.cpp
        Tracker/graph/GTL/src/debug.cpp
        Tracker/graph/GTL/src/dfs.cpp
        Tracker/graph/GTL/src/dijkstra.cpp
        Tracker/graph/GTL/src/edge.cpp
        Tracker/graph/GTL/src/embedding.cpp
        Tracker/graph/GTL/src/fm_partition.cpp
        Tracker/graph/GTL/src/gml_parser.cpp
        Tracker/graph/GTL/src/gml_scanner.cpp
        Tracker/graph/GTL/src/graph.cpp
        Tracker/graph/GTL/src/maxflow_ff.cpp
        Tracker/graph/GTL/src/maxflow_pp.cpp
        Tracker/graph/GTL/src/maxflow_sap.cpp
        Tracker/graph/GTL/src/min_tree.cpp
        Tracker/graph/GTL/src/node.cpp
        Tracker/graph/GTL/src/planarity.cpp
        Tracker/graph/GTL/src/pq_node.cpp
        Tracker/graph/GTL/src/pq_tree.cpp
        Tracker/graph/GTL/src/ratio_cut_partition.cpp
        Tracker/graph/GTL/src/st_number.cpp
        Tracker/graph/GTL/src/topsort.cpp
        )

set(gtl_header
        Tracker/graph/GTL/include/GTL/algorithm.h
        Tracker/graph/GTL/include/GTL/bellman_ford.h
        Tracker/graph/GTL/include/GTL/bfs.h
        Tracker/graph/GTL/include/GTL/biconnectivity.h
        Tracker/graph/GTL/include/GTL/bid_dijkstra.h
        Tracker/graph/GTL/include/GTL/bin_heap.h
        Tracker/graph/GTL/include/GTL/components.h
        Tracker/graph/GTL/include/GTL/debug.h
        Tracker/graph/GTL/include/GTL/dfs.h
        Tracker/graph/GTL/include/GTL/dijkstra.h
        Tracker/graph/GTL/include/GTL/edge.h
        Tracker/graph/GTL/include/GTL/edge_data.h
        Tracker/graph/GTL/include/GTL/edge_map.h
        Tracker/graph/GTL/include/GTL/embedding.h
        Tracker/graph/GTL/include/GTL/fm_partition.h
        Tracker/graph/GTL/include/GTL/gml_parser.h
        Tracker/graph/GTL/include/GTL/gml_scanner.h
        Tracker/graph/GTL/include/GTL/graph.h
        Tracker/graph/GTL/include/GTL/maxflow_ff.h
        Tracker/graph/GTL/include/GTL/maxflow_pp.h
        Tracker/graph/GTL/include/GTL/maxflow_sap.h
        Tracker/graph/GTL/include/GTL/min_tree.h
        Tracker/graph/GTL/include/GTL/ne_map.h
        Tracker/graph/GTL/include/GTL/node.h
        Tracker/graph/GTL/include/GTL/node_data.h
        Tracker/graph/GTL/include/GTL/node_map.h
        Tracker/graph/GTL/include/GTL/planarity.h
        Tracker/graph/GTL/include/GTL/pq_node.h
        Tracker/graph/GTL/include/GTL/pq_tree.h
        Tracker/graph/GTL/include/GTL/ratio_cut_partition.h
        Tracker/graph/GTL/include/GTL/st_number.h
        Tracker/graph/GTL/include/GTL/symlist.h
        Tracker/graph/GTL/include/GTL/topsort.h
        Tracker/graph/GTL/include/GTL/version.h
        Tracker/graph/GTL/include/GTL/GTL.h
        )

set(folder_headers
        Tracker/Ctracker.h
        Tracker/track.h
        Tracker/HungarianAlg/HungarianAlg.h
        Tracker/LocalTracker.h
        Tracker/Kalman.h
        )
set(folder_source
        Tracker/Ctracker.cpp
        Tracker/track.cpp
        Tracker/HungarianAlg/HungarianAlg.cpp
        Tracker/LocalTracker.cpp
        Tracker/Kalman.cpp
        )
set(yolo_source
        yolo/activation_kernels.cu
        yolo/blas_kernels.cu
        yolo/box.cpp
        yolo/cuda.cpp
        yolo/detector.cpp
        yolo/image_opencv.cpp
        yolo/image.cpp
        yolo/yolo_layer.cpp
        )
set(yolo_headers
        yolo/activations.h
        yolo/blas.h
        yolo/box.h
        yolo/cuda.h
        yolo/detector.h
        yolo/image_opencv.h
        yolo/image.h
        yolo/yolo_layer.h
        )

add_executable(${PROJECT_NAME}
        src/yolo_main.cpp 
        include/SimplePipeline.h
        include/defines.h
        include/nms.h
        ${folder_source} 
        ${folder_headers} 
        ${graph_source} 
        ${graph_header} 
        ${gtl_source} 
        ${gtl_header}
        ${yolo_source}
        ${yolo_headers}
)

SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_headers})

SOURCE_GROUP("graph" FILES ${graph_source} ${graph_header})
SOURCE_GROUP("GTL" FILES ${gtl_source} ${gtl_header})
SOURCE_GROUP("yolo" FILES ${yolo_source} ${yolo_headers})

# ----------------------------------------------------------------------------
# Link libraries to project
# ----------------------------------------------------------------------------
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Caffe_LIBRARIES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES(${PROJECT_NAME} /usr/local/nvidia/lib64/libcudart.so.9.1)