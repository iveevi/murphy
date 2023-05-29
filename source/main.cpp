#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <omp.h>

#include <glm/glm.hpp>
#include <glm/gtx/closest_point.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <gif-h/gif.h>

#include "gl.hpp"
#include "image.hpp"
#include "raster.hpp"
#include "grad.hpp"
#include "optimizer.hpp"
#include "render.hpp"

// Sliding window for average and variance
template <size_t N>
struct SlidingWindow {
        float window[N];
        int index;
        bool full;

        float mean;
        float variance;

        SlidingWindow() : index { 0 }, full { false }, mean { 0.0f }, variance { 0.0f } {
                for (int i = 0; i < N; i++)
                        window[i] = 0.0f;
        }

        void push(float value) {
                float old = window[index];
                float old_mean = mean;
                window[index] = value;

                index = (index + 1) % N;
                full = full || index == 0;
                int size = full ? N : index;

                mean += (value - old) / N;
                variance += (value - old) * (value - mean + old - old_mean) / N;
        }
};

int main(int argc, char *argv[])
{
        // TODO: option for image -> image or circle -> image
        std::string path = "../duck.jpg";
        bool invert = false;

        for (int i = 0; i < argc; i++) {
                if (std::string(argv[i]) == "--invert" || std::string(argv[i]) == "-i")
                        invert = true;
                else if (std::string(argv[i]) == "--path" || std::string(argv[i]) == "-p")
                        path = std::string(argv[i + 1]);
        }

        // Load target image
        int width, height, channels;
        stbi_set_flip_vertically_on_load(true);
        unsigned char *data = stbi_load(path.c_str(), &width, &height, &channels, 0);

        Image <SingleChannel> target_interior { width, height };
        Image <ColorChannel> target_display{ width, height };

        for (int i = 0; i < width * height; i++) {
                float r = data[channels * i + 0] / 255.0f;
                float g = data[channels * i + 1] / 255.0f;
                float b = data[channels * i + 2] / 255.0f;
                float a = channels == 4 ? data[channels * i + 3] / 255.0f : 1.0f;
                // TODO: alpha as flag (default on)

                float L = a * (0.2126f * r + 0.7152f * g + 0.0722f * b);

                int x = i % width;
                int y = i / width;

                // TODO: args for alpha smooth cutoff
                target_interior(x, y) = invert ? 1.0f - L : L;
                target_interior(x, y) = glm::smoothstep(0.0f, 0.2f, target_interior(x, y));

                // TODO: color alpha
                target_display(x, y) = { r, g, b };
        }

        Image <SingleChannel> target_boundary = sobel(target_interior);

        float aspect = (float) width / (float) height;
        WINDOW_WIDTH *= aspect;

        target_display.resize(WINDOW_WIDTH, WINDOW_HEIGHT);

        if (!data) {
                fprintf(stderr, "Failed to load image: %s\n", path.c_str());
                return -1;
        }
        // TODO: sobel to get boundary

        printf("Loaded image: %s (%dx%d, %d channels)\n", path.c_str(), width, height, channels);

        // Initialize GLFW
        GLFWwindow *window = glfw_init();
        if (!window) {
                fprintf(stderr, "Failed to initialize GLFW\n");
                return -1;
        }

        // Create initial curve
        Curve curve;
        float radius = 0.5;
        for (float theta = 0.0f; theta < 2.0f * M_PI; theta += 0.01f)
                curve.push_back(glm::vec2 { radius * cos(theta), radius * sin(theta) });

        printf("Curve elements size: %lu\n", curve.size());
        CurveRasterizer source { curve };

        // Allocate images for rendering
        Image <SingleChannel> ann_grad { WINDOW_WIDTH, WINDOW_HEIGHT };
        Image <SingleChannel> ann_normal { WINDOW_WIDTH, WINDOW_HEIGHT };
        Image <ColorChannel> final { WINDOW_WIDTH, WINDOW_HEIGHT };

        // Prepare GIF writing
        GifWriter gif;
        GifBegin(&gif, "out.gif", WINDOW_WIDTH, WINDOW_HEIGHT, 2);
        uint8_t *frame = new uint8_t[WINDOW_WIDTH * WINDOW_HEIGHT * 4];

        int power = 100;
        bool show_target_alpha = false;
        bool show_normal = true;
        bool show_gradient = true;

        bool pause = false;
        bool pause_rise = false;

        std::map <int, bool> pressed;

        int iterations = 0;
        SlidingWindow <50> iou_window;
        Adam opt = Adam { 0.01f };

        // Initialize ImGui
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        ImGui::StyleColorsDark();

        // TODO: max number of iterations
        // TODO: ImGui
        while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                        glfwSetWindowShouldClose(window, GL_TRUE);

                if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !pause_rise) {
                        pause = !pause;
                        pause_rise = true;
                } else if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) {
                        pause_rise = false;
                }

                if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS && !pressed[GLFW_KEY_T]) {
                        show_target_alpha = !show_target_alpha;
                        pressed[GLFW_KEY_T] = true;
                } else if (glfwGetKey(window, GLFW_KEY_T) == GLFW_RELEASE) {
                        pressed[GLFW_KEY_T] = false;
                }

                if (pause)
                        continue;

                // Render here
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                // Compute gradients and update
                final.clear(glm::vec3 {1.0f});

                source.clear();
                ann_grad.clear();
                ann_normal.clear();

                source.rasterize();
                // auto derivatives = source.grad(target.interior, 5);
                auto derivatives = source.grad(target_interior, 5);
                auto normals = source.normal();

                laplacian_diffuse(derivatives, power);

                std::vector <glm::vec2> ann_points;
                std::vector <glm::vec2> ann_derivatives;
                std::vector <glm::vec2> ann_normals;

                for (int i = 0; i < source.curve.size(); i += 10) {
                        ann_points.push_back(source.curve[i]);
                        ann_derivatives.push_back(derivatives[i]);
                        ann_normals.push_back(0.02f * normals[i]);
                }

                std::vector <LineSegment> items;
                if (show_normal)
                        quiver(ann_normal, ann_points, ann_normals);

                if (show_gradient)
                        quiver(ann_grad, ann_points, ann_derivatives);

                float grad_avg = opt.step(source.curve, derivatives);
                source.update();

                // Composite and draw
                std::vector <const Image <SingleChannel> *> images;
                std::vector <glm::vec3> colors;
                std::vector <float> alphas;

                images.push_back(&source.boundary);
                colors.push_back({ 0.0f, 1.0f, 1.0f });
                alphas.push_back(1.0f);

                images.push_back(&source.interior);
                colors.push_back({ 0.4f, 0.5f, 0.7f });
                alphas.push_back(0.4f);

                images.push_back(&ann_grad);
                colors.push_back({ 1.0f, 0.2f, 0.2f });
                alphas.push_back(1.0f);

                images.push_back(&ann_normal);
                colors.push_back({ 0.2f, 1.0f, 0.2f });
                alphas.push_back(1.0f);

                if (show_target_alpha) {
                        images.push_back(&target_interior);
                        colors.push_back({ 0.7f, 0.8f, 0.2f });
                        alphas.push_back(0.5f);
                }

                if (show_target_alpha)
                        composite(final, images, colors, alphas);
                else
                        composite(final, target_display, images, colors, alphas);

                glDrawPixels(final.width, final.height, ColorChannel::gl, GL_FLOAT, final.data);

                // Write frame to GIF
                convert(final, frame);
                GifWriteFrame(&gif, frame, WINDOW_WIDTH, WINDOW_HEIGHT, 2);

                // Print info
                float metric = iou(target_interior, source.interior);
                // printf("\033[2J\033[1;1H");
                // printf("IOU: %f\n", metric);

                if (metric > 0.85)
                        power = 20; // TODO: also depends on the curve size (and
                // variance of grdients)
                if (metric > 0.95)
                        power = 5;

                // Subdivide the curve if needed
                iou_window.push(100.0f * metric);

                // ImGui interface
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();

                std::filesystem::path target_path = path;

                ImGui::NewFrame();
                ImGui::Begin("Murphy");
                ImGui::Text("Path: %s", target_path.filename().c_str());
                ImGui::Text("IOU: %.2f%% (var=%.2f)", iou_window.mean, iou_window.variance);
                ImGui::Text("Gradient average: %f", grad_avg);
                ImGui::Text("Power: %d", power);
                ImGui::Text("Iterations: %d", iterations);
                ImGui::Text("Curve size: %lu", source.curve.size());
                ImGui::Text("Time: %f ms", ImGui::GetIO().DeltaTime * 1000.0f);
                ImGui::End();

                ImGui::Render();
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                // Subdivide the curve if needed
                if ((++iterations) % 50 == 0) {
                        printf("Subdividing, gradient average: %f\n", grad_avg);
                        auto new_indices = source.adaptive_subdivide(derivatives);
                        opt.upscale(source.curve, new_indices);
                }

                // Swap front and back buffers
                glfwSwapBuffers(window);
        }

        // Finish GIF writing
        GifEnd(&gif);
}
