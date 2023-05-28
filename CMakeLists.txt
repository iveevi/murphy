project(curvature-ir C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(curves main.cpp glad/src/glad.c)
target_link_libraries(curves glfw OpenMP::OpenMP_CXX)
include_directories(glad/include glm)
