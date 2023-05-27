#include <iostream>
#include <vector>
#include <algorithm>

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
        static constexpr float clear = 0.0f;
};

struct ColorChannel {
        using type = glm::vec3;
        static constexpr int gl = GL_RGB;
        static constexpr int channels = 3;
        static constexpr glm::vec3 clear = glm::vec3 { 0.0f };
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
                data = new typename Channel::type[width * height * Channel::channels] { Channel::clear };
        }

        // Move only
        Image(const Image &) = delete;
        Image(Image &&other) : width(other.width), height(other.height), data(other.data) {
                other.data = nullptr;
        }

        Image &operator=(const Image &) = delete;
        Image &operator=(Image &&other) {
                width = other.width;
                height = other.height;
                data = other.data;
                other.data = nullptr;
                return *this;
        }

        ~Image() {
                delete[] data;
        }

        typename Channel::type &operator()(int x, int y) {
                return data[y * width + x];
        }
        
        const typename Channel::type &operator()(int x, int y) const {
                return data[y * width + x];
        }

        typename Channel::type *operator[](int y) {
                return data + y * width;
        }
        
        const typename Channel::type *operator[](int y) const {
                return data + y * width;
        }

        void clear(typename Channel::type val = typename Channel::type { 0 }) {
                for (int i = 0; i < width * height * Channel::channels; i++)
                        data[i] = val;
        }
};

// Render viewport configuration
struct {
        float aspect = float(WINDOW_WIDTH) / WINDOW_HEIGHT;
        // TODO: ranges as vec2
        float yrange = 1.0f; // Y from -1 to 1
        float xrange = yrange * aspect;
        float epsilon = std::min(xrange, yrange)/500.0f;

        // Cartesian to uv coordinates
        glm::vec2 to_uv(const glm::vec2 &point) {
                return (point + glm::vec2 { xrange, yrange }) / (2.0f * glm::vec2 { xrange, yrange });
        }

        float to_uv_x(float x) {
                return (x + xrange) / (2.0f * xrange);
        }

        float to_cartesian_y(float y) {
                return 2 * y * yrange - yrange;
        }
} viewport;

// Rasterizing lines
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
void rasterize_boundary(Image <SingleChannel> &img, const LineSegment &line)
{
        float thickness = 2.0f * viewport.epsilon;
        glm::vec2 line_min = viewport.to_uv(glm::min(line.v0, line.v1) - thickness);
        glm::vec2 line_max = viewport.to_uv(glm::max(line.v0, line.v1) + thickness);

        int min_x = std::max(0, int(line_min.x * img.width));
        int min_y = std::max(0, int(line_min.y * img.height));
        int max_x = std::min(img.width, int(line_max.x * img.width));
        int max_y = std::min(img.height, int(line_max.y * img.height));

        // TODO: line search from min y to max y (including thickness)
        for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                        // Project point
                        glm::vec2 uv = glm::vec2 { x, y } / glm::vec2 { img.width, img.height };
                        glm::vec2 point = glm::vec2 { viewport.xrange, viewport.yrange } * (uv - 0.5f) * 2.0f;

                        // TODO: smoothstep
                        float dist = distance(line, point);
                        if (dist < thickness) {
                                // Fall off after epsilon
                                float c = 1.0f - glm::smoothstep(0.0f, thickness, dist);
                                img(x, y) = std::max(img(x, y), c);
                        }
                }
        }
}

using Curve = std::vector <glm::vec2>;

std::pair <glm::vec2, glm::vec2> bounds(const Curve &curve)
{
        glm::vec2 min { std::numeric_limits <float>::max() };
        glm::vec2 max { std::numeric_limits <float>::min() };

        for (const auto &point : curve) {
                min = glm::min(min, point);
                max = glm::max(max, point);
        }

        return { min, max };
}

// Rasterize interior of boundary from MIN to MAX (both Cartesian coordinates)
void rasterize_interior(Image <SingleChannel> &dst, const Curve &curve, const glm::vec2 &min, const glm::vec2 &max)
{
        // Use analytical horizontal line scan
        float thickness = 2.0f * viewport.epsilon;
        glm::vec2 min_uv = viewport.to_uv(min - thickness);
        glm::vec2 max_uv = viewport.to_uv(max + thickness);

        int min_x = std::max(0, int(min_uv.x * dst.width));
        int min_y = std::max(0, int(min_uv.y * dst.height));
        int max_x = std::min(dst.width, int(max_uv.x * dst.width));
        int max_y = std::min(dst.height, int(max_uv.y * dst.height));

        for (int y = min_y; y <= max_y; y++) {
                // Cartesian vertical component
                float cy = viewport.to_cartesian_y(float(y)/dst.height);

                // Check whether (and when) the curve intersects the horizontal line
                std::vector <int32_t> intersections;
                for (int i = 0; i < curve.size(); i++) {
                        int j = (i + 1) % curve.size();
                        const glm::vec2 &p0 = curve[i];
                        const glm::vec2 &p1 = curve[j];

                        // Skip if the line is horizontal
                        if (p0.y == p1.y)
                                continue;

                        // Skip if the line does not intersect the horizontal line
                        if (p0.y < cy && p1.y < cy)
                                continue;
                        if (p0.y > cy && p1.y > cy)
                                continue;

                        // Find the intersection
                        float t = (cy - p0.y) / (p1.y - p0.y);
                        float cx = p0.x + t * (p1.x - p0.x);

                        // Skip if the intersection is outside the viewport
                        if (cx < min.x || cx > max.x)
                                continue;

                        // Convert to screen coordinates
                        // TODO: smooth interpolation?
                        intersections.push_back(viewport.to_uv_x(cx) * dst.width);
                }

                // Sort and fill
                std::sort(intersections.begin(), intersections.end());

                for (int i = 0; i < intersections.size(); i += 2) {
                        if (i + 1 >= intersections.size())
                                break;

                        int x0 = std::max(min_x, intersections[i]);
                        int x1 = std::min(max_x, intersections[i + 1]);

                        for (int x = x0; x <= x1; x++)
                                dst(x, y) = 1.0f;
                }
        }
}

// Compositing images
void composite(Image <ColorChannel> &dst, const std::vector <Image <SingleChannel>> &srcs, const std::vector <glm::vec3> &colors)
{
        assert(srcs.size() == colors.size());
        for (int i = 0; i < srcs.size(); i++)
                assert(srcs[i].width == dst.width && srcs[i].height == dst.height);

        // Use src images as the alpha (and use a weighted average)
        for (int y = 0; y < dst.height; y++) {
                for (int x = 0; x < dst.width; x++) {
                        glm::vec3 color { 0.0f };
                        float alpha = 0.0f;

                        for (int i = 0; i < srcs.size(); i++) {
                                float a = srcs[i](x, y);
                                color += a * colors[i];
                                alpha += a;
                        }

                        // if (alpha > 0.0f)
                        //         color /= alpha;

                        dst(x, y) = color;
                }
        }
}

int main()
{
        GLFWwindow *window = glfw_init();
        if (!window) {
                fprintf(stderr, "Failed to initialize GLFW\n");
                return -1;
        }

        Image <SingleChannel> boundary { WINDOW_WIDTH, WINDOW_HEIGHT };
        Image <SingleChannel> interior { WINDOW_WIDTH, WINDOW_HEIGHT };

        boundary.clear();
        interior.clear();

        Curve curve;
       
        // TODO: generating random closed curves (non intersecting)

        float radius = 0.5;
        for (float theta = 0.0f; theta < 2.0f * M_PI; theta += 0.01f) {
                float r = radius;
                for (int n = 0; n < 5; n++)
                        r += 0.4 * pow(0.5f, n) * sin(pow(2.0f, n) * theta);
                r = std::max(0.2f * radius, r);
                
                curve.push_back(glm::vec2 { r * cos(theta), r * sin(theta) });
        }

        printf("Curve elements size: %lu\n", curve.size());

        for (int i = 0; i < curve.size(); i++) {
                int j = (i + 1) % curve.size();
                rasterize_boundary(boundary, LineSegment { curve[i], curve[j] });
        }

        auto [min, max] = bounds(curve);
        // rasterize_interior(interior, boundary, min, max);
        rasterize_interior(interior, curve, min, max);

        Image <ColorChannel> final { WINDOW_WIDTH, WINDOW_HEIGHT };

        std::vector <Image <SingleChannel>> srcs; // { std::move(boundary), std::move(interior) };
        srcs.emplace_back(std::move(boundary));
        srcs.emplace_back(std::move(interior));

        std::vector <glm::vec3> colors { { 1.0f, 0.7f, 0.0f }, { 1.0f, 0.7f, 0.0f } };
        composite(final, srcs, colors);

        while (!glfwWindowShouldClose(window)) {
                // Render here
                glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                // Draw image
                glDrawPixels(final.width, final.height, ColorChannel::gl, GL_FLOAT, final.data);

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
