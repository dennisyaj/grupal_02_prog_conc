cmake_minimum_required(VERSION 3.25)
project(grupal_02_prog_conc)

set(CMAKE_CXX_STANDARD 17)

include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#10.0.0")

find_package(MPI REQUIRED)

message(STATUS "*******Run: ${MPIEXEC}")

add_executable(grupal_02_prog_conc main.cpp)
target_link_libraries(grupal_02_prog_conc fmt::fmt MPI::MPI_CXX)

