cmake_minimum_required (VERSION 2.8.11)
project(lin_alg_tools)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message("-- Build type: Release")
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LAPACK REQUIRED)

include_directories(
  include 
  ${GTEST_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(lin_alg_tools INTERFACE)
target_link_libraries(lin_alg_tools INTERFACE ${LAPACK_LIBRARIES})
target_include_directories(lin_alg_tools INTERFACE include)

add_executable(lin_alg_tools_test src/test.cpp)
target_link_libraries(lin_alg_tools_test ${GTEST_LIBRARIES} gtest gtest_main pthread lin_alg_tools)
