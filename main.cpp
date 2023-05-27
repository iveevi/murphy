#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtx/closest_point.hpp>

constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;

GLFWwindow *glfw_init();

struct SingleChannel {
        using type = float;
        static constexpr int gl = GL_RED;
        static constexpr int channels = 1;
};

// Fixed channel image
template <typename Channel>
struct Image {
        int width;
        int height;
        typename Channel::type *data;

        static constexpr int gl = Channel::gl;
        static constexpr int channels = Channel::channels;

        Image(int w, int h) : width(w), height(h) {
                data = new float[width * height * Channel::channels]
                        { typename Channel::type { 0 } };
        }

        ~Image() {
                delete[] data;
        }

        typename Channel::type &operator()(int x, int y) {
                return data[y * width + x];
        }

        void clear(typename Channel::type val = typename Channel::type { 0 }) {
                for (int i = 0; i < width * height * Channel::channels; i++)
                        data[i] = val;
        }
};

struct LineSegment {
        glm::vec2 v0;
        glm::vec2 v1;
};

// Closest distance from a point to a line segment
float distance(const LineSegment &line, const glm::vec2 &point)
{
        return glm::length(glm::closestPointOnLine(point, line.v0, line.v1) - point);
}

// TODO: thickness
void rasterize(Image <SingleChannel> &img, const LineSegment &line)
{
        constexpr float aspect = float(WINDOW_WIDTH) / WINDOW_HEIGHT;
        constexpr float yrange = 1.0f; // Y from -1 to 1
        constexpr float xrange = yrange * aspect;
        constexpr float epsilon = std::min(xrange, yrange)/100.0f;

        // printf("Line: (%f, %f) -> (%f, %f)\n", line.v0.x, line.v0.y, line.v1.x, line.v1.y);

        glm::vec2 line_min = glm::min(line.v0, line.v1) - 2.0f * glm::vec2 { epsilon } + glm::vec2 { xrange, yrange };
        line_min = line_min / (2.0f * glm::vec2 { xrange, yrange });

        glm::vec2 line_max = glm::max(line.v0, line.v1) + 2.0f * glm::vec2 { epsilon } + glm::vec2 { xrange, yrange };
        line_max = line_max / (2.0f * glm::vec2 { xrange, yrange });

        int min_x = std::max(0, int(line_min.x * img.width));
        int min_y = std::max(0, int(line_min.y * img.height));
        int max_x = std::min(img.width, int(line_max.x * img.width));
        int max_y = std::min(img.height, int(line_max.y * img.height));

        // printf("Range: (%d, %d) -> (%d, %d)\n", min_x, min_y, max_x, max_y);

        // for (int y = 0; y < img.height; y++) {
        //         for (int x = 0; x < img.width; x++) {
        for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                        // Project point
                        glm::vec2 uv = glm::vec2 { x, y } / glm::vec2 { img.width, img.height };
                        glm::vec2 point = glm::vec2 { xrange, yrange } * (uv - 0.5f) * 2.0f;

                        // TODO: smoothstep
                        float dist = distance(line, point);
                        if (dist < 2 * epsilon) {
                                // Fall off after epsilon
                                float c = 1.0f - glm::smoothstep(0.0f, 2 * epsilon, dist);
                                img(x, y) = std::max(img(x, y), c);
                        }
                }
        }
}

using Curve = std::vector <glm::vec2>;

int main()
{
        GLFWwindow *window = glfw_init();
        if (!window) {
                fprintf(stderr, "Failed to initialize GLFW\n");
                return -1;
        }

        Image <SingleChannel> img { WINDOW_WIDTH, WINDOW_HEIGHT };

        img.clear();
        rasterize(img, LineSegment { glm::vec2 { 0.1f, 0.1f }, glm::vec2 { 0.9f, 0.9f } });

        float radius = 0.5;

        Curve curve;
        for (float theta = 0.0f; theta < 2.0f * M_PI; theta += 0.1f)
                curve.push_back(glm::vec2 { radius * cos(theta), radius * sin(theta) });

        printf("Curve size: %lu\n", curve.size());

        for (int i = 0; i < curve.size(); i++) {
                int j = (i + 1) % curve.size();
                rasterize(img, LineSegment { curve[i], curve[j] });
        }

        while (!glfwWindowShouldClose(window)) {
                // Render here
                glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                // Draw image
                glDrawPixels(img.width, img.height, GL_RED, GL_FLOAT, img.data);

                // Swap front and back buffers
                glfwSwapBuffers(window);

                // Poll for and process events
                glfwPollEvents();
        }
}

GLFWwindow *glfw_init()
{
	// Basic window
	GLFWwindow *window = nullptr;
	if (!glfwInit())
		return nullptr;

	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "SDF Engine", NULL, NULL);

	// Check if window was created
	if (!window) {
		glfwTerminate();
		return nullptr;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Load OpenGL functions using GLAD
	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
		fprintf(stderr, "Failed to initialize OpenGL context\n");
		return nullptr;
	}

	// Set up callbacks
	// glfwSetCursorPosCallback(window, mouse_callback);
	// glfwSetMouseButtonCallback(window, mouse_button_callback);
	// glfwSetKeyCallback(window, keyboard_callback);

	const GLubyte* renderer = glGetString(GL_RENDERER);
	printf("Renderer: %s\n", renderer);

	return window;
}
