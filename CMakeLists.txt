project(curvature-ir C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(curves
        source/main.cpp
        source/gl.cpp

        vendor/glad/src/glad.c

        vendor/imgui/imgui.cpp
        vendor/imgui/imgui_demo.cpp
        vendor/imgui/imgui_draw.cpp
        vendor/imgui/imgui_widgets.cpp
        vendor/imgui/imgui_tables.cpp
        vendor/imgui/backends/imgui_impl_glfw.cpp
        vendor/imgui/backends/imgui_impl_opengl3.cpp
)

target_link_libraries(curves glfw OpenMP::OpenMP_CXX)
include_directories(. vendor vendor/glad/include vendor/glm vendor/imgui)
