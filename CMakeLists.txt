cmake_minimum_required(VERSION 2.8.3)
project(covis)

add_definitions(-std=c++11)
#set(CMAKE_CXX_FLAGS "-std=c++11)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -Wall -pthread") # -Wextra -Werror
set(CMAKE_BUILD_TYPE "RELEASE")


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdPartLib/g2o/cmake_modules)
set(G2O_ROOT /usr/local/include/g2o)
find_package(G2O REQUIRED) 

find_package (OpenCV 3 REQUIRED)
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
find_package (Eigen3 REQUIRED)

find_package (CSparse REQUIRED )
find_package (Sophus REQUIRED )
find_package (yaml-cpp REQUIRED )

# pcl
#find_package( PCL REQUIRED)
#include_directories( ${PCL_INCLUDE_DIRS} )
#add_definitions( ${PCL_DEFINITIONS} )

#FIND_PACKAGE(octomap REQUIRED )
#FIND_PACKAGE(octovis REQUIRED )
#INCLUDE_DIRECTORIES(${OCTOMAP_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS
    nodelet
    roscpp
    rostime
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    pcl_ros
    tf
    visualization_msgs
    image_transport
    cv_bridge
    message_generation
    message_filters
    )

add_message_files(
    FILES
    KeyFrame.msg
    CorrectionInf.msg
    )

generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    )

## Declare a catkin package
catkin_package(
    CATKIN_DEPENDS message_runtime
    )


include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdPartLib/DBow2/"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdPartLib/DLib/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/processing/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/visualization/"
    #"${CMAKE_CURRENT_SOURCE_DIR}/src/octofeeder/"
    
    )


SET(G2O_LIBS cholmod cxsparse -lg2o_cli -lg2o_core
    -lg2o_csparse_extension -lg2o_ext_freeglut_minimal -lg2o_incremental
    -lg2o_interactive -lg2o_interface -lg2o_opengl_helper -lg2o_parser
    -lg2o_simulator -lg2o_solver_cholmod -lg2o_solver_csparse
    -lg2o_solver_dense -lg2o_solver_pcg -lg2o_solver_slam2d_linear
    -lg2o_solver_structure_only -lg2o_stuff -lg2o_types_data -lg2o_types_icp
    -lg2o_types_sba -lg2o_types_sclam2d -lg2o_types_sim3 -lg2o_types_slam2d
    -lg2o_types_slam3d)

## Declare a C++ library
add_library(covis
    #processing
    src/processing/feature_dem.cpp
    src/processing/depth_camera.cpp
    src/processing/landmark.cpp
    src/processing/camera_frame.cpp
    src/processing/triangulation.cpp
    src/processing/lkorb_tracking.cpp
    src/processing/imu_state.cpp
    src/processing/vi_motion.cpp
    src/processing/optimize_in_frame.cpp
    #vis
    src/visualization/rviz_frame.cpp
    src/visualization/rviz_path.cpp
    src/visualization/rviz_pose.cpp
    src/visualization/rviz_odom.cpp
    src/visualization/rviz_edge.cpp
    src/visualization/rviz_mesh.cpp
    #msg
    src/utils/keyframe_msg_handler.cpp
    src/utils/correction_inf_msg.cpp

    #node tracking
    src/frontend/vo_tracking.cpp
    src/frontend/f2f_tracking.cpp
    #node localmap
    #src/backend/vo_localmap.cpp

    src/backend/poselmbag.cpp

    #node map merging
    src/backend/vo_merging.cpp
    src/backend/merging.cpp
    #src/octofeeder/octomap_feeder.cpp

    #brief pattern
    3rdPartLib/DLib/DUtils/Random.cpp
    3rdPartLib/DLib/DUtils/Timestamp.cpp
    3rdPartLib/DLib/DVision/BRIEF.cpp
    3rdPartLib/DBow2/DBoW/BowVector.cpp
    3rdPartLib/DBow2/DBoW/FBrief.cpp
    3rdPartLib/DBow2/DBoW/FeatureVector.cpp
    3rdPartLib/DBow2/DBoW/QueryResults.cpp
    3rdPartLib/DBow2/DBoW/ScoringObject.cpp
    3rdPartLib/DBow2/VocabularyBinary.cpp
    )

add_dependencies(covis
    covis_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS})

target_link_libraries(covis
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${Sophus_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${G2O_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_SYSTEM_LIBRARY}
    #${OCTOMAP_LIBRARIES}
    )

#independent modules
#1 euroc_publisher publish path
add_executable(vo_repub_rec
    src/independ_modules/vo_repub_rec.cpp)
target_link_libraries(vo_repub_rec
    ${catkin_LIBRARIES}
    ${Sophus_LIBRARIES})

add_executable(kitti_publisher
    src/independ_modules/kitti_publisher.cpp
    src/visualization/rviz_path.cpp)
target_link_libraries(kitti_publisher
    ${catkin_LIBRARIES}
    ${Sophus_LIBRARIES})

#add_executable(create
#    src/test/create_voc.cpp
#)
#target_link_libraries(create
#    ${OpenCV_LIBRARIES}
    #${DBoW3_LIBRARIES}
#)
add_executable(se3
    src/test/se3.cpp
    src/visualization/rviz_pose.cpp)
target_link_libraries(se3
    ${OpenCV_LIBRARIES}
    ${G2O_LIBS}
    ${catkin_LIBRARIES}
    ${Sophus_LIBRARIES})

