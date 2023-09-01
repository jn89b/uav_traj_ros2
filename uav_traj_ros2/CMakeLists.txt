cmake_minimum_required(VERSION 3.5)

#change name of package to this
project(uav_traj_ros2)
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#https://docs.ros.org/en/foxy/How-To-Guides/Ament-CMake-Python-Documentation.html

# Find dependencies this will allow CPP and python
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(drone_interfaces REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# messages
# rosidl_generate_interfaces(${PROJECT_NAME}
#   # "msg/WayPoints.msg"
#   DEPENDENCIES
#   builtin_interfaces 
#   geometry_msgs 
#   std_msgs 
#   rosidl_default_generators 
#  )

# services
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "srv/CapitalFullName.srv"
# )

# ----------------- CPP ------------------------------------ 
#Include Cpp "include" directory
include_directories(include)

# install(
# TARGETS ${PROJECT_NAME}
# EXPORT export_${PROJECT_NAME}
# LIBRARY DESTINATION lib/${PROJECT_NAME}
# ARCHIVE DESTINATION lib/${PROJECT_NAME}
# RUNTIME DESTINATION bin/${PROJECT_NAME}
# INCLUDES DESTINATION include
# )

# Create a sources variable with a link to all cpp files to compile
set(SOURCES
    src/Astar.cpp
    src/GridMap.cpp
    src/PositionVector.cpp
)

set(MIN_EXAMPLE
    src/publisher_library.cpp)

#executables For cpp
add_executable(sas_main ${SOURCES} scripts/main_sas.cpp)
install(TARGETS
  sas_main
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(main_node ${MIN_EXAMPLE} scripts/use_library.cpp) 
target_include_directories(main_node PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${rclcpp_INCLUDE_DIRS}
)
ament_target_dependencies(
  main_node
  "rclcpp"
)
install(TARGETS main_node
  DESTINATION lib/${PROJECT_NAME})



set(PLANNER
  src/Astar.cpp
  src/GridMap.cpp
  src/PositionVector.cpp
  src/GlobalPlanner.cpp
)

# This is the global planner node
add_executable(global_planner_node ${PLANNER} scripts/global_planner_node.cpp)
target_include_directories(global_planner_node PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${rclcpp_INCLUDE_DIRS}
)
ament_target_dependencies(
  global_planner_node
  "rclcpp"
  drone_interfaces
)
install(TARGETS global_planner_node
  DESTINATION lib/${PROJECT_NAME})

# ----------------- Python ------------------------------------
# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/waypoint_viz.py
  scripts/fw_mpc_node.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()

# ----------------- Testing ------------------------------------
# if(BUILD_TESTING)
#   find_package(ament_cmake_pytest REQUIRED)
#   set(_pytest_tests
#     tests/test_a.py
#     # Add other test files here
#   )
#   foreach(_test_path ${_pytest_tests})
#     get_filename_component(_test_name ${_test_path} NAME_WE)
#     ament_add_pytest_test(${_test_name} ${_test_path}
#       APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
#       TIMEOUT 60
#       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#     )
#   endforeach()
# endif()

#launch
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)