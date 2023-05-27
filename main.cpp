#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <tuple>
#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtx/closest_point.hpp>
#include <glm/gtx/rotate_vector.hpp>

constexpr int WINDOW_WIDTH = 750;
constexpr int WINDOW_HEIGHT = 750;

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

        glm::vec2 to_screen(const glm::vec2 &point) {
                return to_uv(point) * glm::vec2 { WINDOW_WIDTH, WINDOW_HEIGHT };
        }

        glm::vec2 to_cartesian(const glm::vec2 &uv) {
                return 2.0f * glm::vec2 { xrange, yrange } * uv - glm::vec2 { xrange, yrange };
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
void rasterize_line(Image <SingleChannel> &img, const LineSegment &line)
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

                        float dist = distance(line, point);
                        if (dist < thickness) {
                                // Fall off after epsilon
                                float c = 1.0f - glm::smoothstep(0.0f, thickness, dist);
                                img(x, y) = std::max(img(x, y), c);
                        }
                }
        }
}

void rasterize_arrow(Image <SingleChannel> &img, const glm::vec2 &origin, const glm::vec2 &direction)
{
        glm::vec2 v0 = origin;
        glm::vec2 v1 = origin + direction;

        glm::vec2 left = -glm::rotate(direction, glm::radians(30.0f))/10.0f;
        glm::vec2 right = -glm::rotate(direction, glm::radians(-30.0f))/10.0f;

        rasterize_line(img, { v0, v1 });
        rasterize_line(img, { v1, v1 + left });
        rasterize_line(img, { v1, v1 + right });
}

void quiver(Image <SingleChannel> &img, const std::vector <glm::vec2> &points, const std::vector <glm::vec2> &directions)
{
        for (int i = 0; i < points.size(); i++)
                rasterize_arrow(img, points[i], directions[i]);
}

using Curve = std::vector <glm::vec2>;

void rasterize_boundary(Image <SingleChannel> &img, const Curve &curve)
{
        for (int i = 0; i < curve.size(); i++) {
                int j = (i + 1) % curve.size();
                rasterize_line(img, { curve[i], curve[j] });
        }
}

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
                std::vector <float> intersections;
                for (int i = 0; i < curve.size(); i++) {
                        int j = (i + 1) % curve.size();
                        const glm::vec2 &p0 = curve[i];
                        const glm::vec2 &p1 = curve[j];

                        // Skip if the line is mostly horizontal
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
                        if (cx < min.x - viewport.epsilon || cx > max.x + viewport.epsilon)
                                continue;

                        // Convert to screen coordinates
                        // TODO: smooth interpolation?
                        intersections.push_back(viewport.to_uv_x(cx));
                }

                // Sort, remove duplicates
                std::sort(intersections.begin(), intersections.end());
                intersections.erase(std::unique(intersections.begin(), intersections.end()), intersections.end());

                for (int i = 0; i < intersections.size(); i += 2) {
                        if (i + 1 >= intersections.size())
                                break;

                        float cx0 = intersections[i];
                        float cx1 = intersections[i + 1];

                        int x0 = std::max(min_x, int(cx0 * dst.width));
                        int x1 = std::min(max_x, int(cx1 * dst.width));

                        for (int x = x0; x <= x1; x++)
                                dst(x, y) = 1.0f;
                }
        }
}

// Compute gradient of line segment using interiors and point sample
std::pair <glm::vec2, glm::vec2> grad_segment(const Image <SingleChannel> &target, const Image <SingleChannel> &interior, const LineSegment &line, float t)
{
        assert(0 <= t && t <= 1.0f);

        // Normal (perpendicular) vector
        glm::vec2 d = glm::normalize(line.v1 - line.v0);
        glm::vec2 n = glm::normalize(glm::vec2(-d.y, d.x));
        glm::vec2 p = line.v0 + t * (line.v1 - line.v0);
        glm::vec2 dp { 0.0f };

        // Get derivative on neighborhood
        static constexpr int N = 3;

        glm::vec2 screen_p = viewport.to_screen(p);

        float w_sum = 0.0f;
        for (int y = -N; y <= N; y++) {
                for (int x = -N; x <= N; x++) {
                        // Ensure valid image coordinates
                        if (screen_p.x + x < 0 || screen_p.x + x >= target.width)
                                continue;

                        if (screen_p.y + y < 0 || screen_p.y + y >= target.height)
                                continue;

                        // Compute weight
                        glm::vec2 q = screen_p + glm::vec2(x, y);
                        glm::vec2 uv = q / glm::vec2(target.width, target.height);
                        glm::vec2 pt = viewport.to_cartesian(uv);

                        // Skip if on the line
                        if (std::fabs(glm::dot(pt - line.v0, line.v1 - line.v0)) < 1e-6f)
                                continue;

                        float w = glm::length(pt - p);

                        float at = target(uv.x * target.width, uv.y * target.height);
                        float ai = interior(uv.x * interior.width, uv.y * interior.height);
                        float delta = at - ai;
                        
                        dp += delta * n * w;
                        w_sum += w;
                }
        }

        if (w_sum < 1e-6f)
                return { glm::vec2 { 0.0f }, glm::vec2 { 0.0f } };

        dp /= w_sum;
        glm::vec2 dv0 = dp * (1.0f - t);
        glm::vec2 dv1 = dp * t;

        return { dv0, dv1 };
}

// Compositing images
void composite(Image <ColorChannel> &dst, const std::vector <const Image <SingleChannel> *> &srcs, const std::vector <glm::vec3> &colors)
{
        assert(srcs.size() == colors.size());
        for (int i = 0; i < srcs.size(); i++)
                assert(srcs[i]->width == dst.width && srcs[i]->height == dst.height);

        // Use src images as the alpha (and use a weighted average)
        for (int y = 0; y < dst.height; y++) {
                for (int x = 0; x < dst.width; x++) {
                        glm::vec3 color { 0.0f };
                        float alpha = 0.0f;

                        for (int i = 0; i < srcs.size(); i++) {
                                float a = (*srcs[i])(x, y);
                                color += a * colors[i];
                                alpha += a;
                        }

                        if (alpha > 0.0f)
                                color /= alpha;

                        dst(x, y) = color;
                }
        }
}

// Gradient regularizers
void laplacian_diffuse(std::vector <glm::vec2> &grad)
{
        for (int i = 0; i < grad.size(); i++) {
                int pi = (i - 1 + grad.size()) % grad.size();
                int ni = (i + 1) % grad.size();

                grad[i] = (grad[pi] + grad[ni]) / 2.0f;
        }
}

struct CurveRasterizer {
        Curve curve;

        glm::vec2 min;
        glm::vec2 max;
        
        Image <SingleChannel> boundary;
        Image <SingleChannel> interior;

        CurveRasterizer(const Curve &curve_)
                        : curve { curve_ },
                        boundary { WINDOW_WIDTH, WINDOW_HEIGHT },
                        interior { WINDOW_WIDTH, WINDOW_HEIGHT } {
                std::tie(min, max) = bounds(curve);
        }

        void clear() {
                boundary.clear();
                interior.clear();
        }

        void rasterize() {
                rasterize_boundary(boundary, curve);
                rasterize_interior(interior, curve, min, max);
        }

        std::vector <glm::vec2> grad(const Image <SingleChannel> &target, int N) {
                std::mt19937 rng { std::random_device {}() };
                std::uniform_real_distribution <float> dist { 0.0f, 1.0f };

                std::vector <glm::vec2> grad { curve.size(), glm::vec2 { 0.0f } };
                for (int i = 0; i < curve.size(); i++) {
                        int j = (i + 1) % curve.size();

                        glm::vec2 p0 = curve[i];
                        glm::vec2 p1 = curve[j];

                        glm::vec2 dv0 { 0.0f };
                        glm::vec2 dv1 { 0.0f };

                        // TODO: monte carlo estimation
                        for (int k = 0; k < N; k++) {
                                float t = dist(rng);
                                auto [sdv0, sdv1] = grad_segment(target, interior, { p0, p1 }, t);
                                
                                dv0 += sdv0;
                                dv1 += sdv1;
                        }
                        
                        dv0 /= N;
                        dv1 /= N;

                        // Account for the fact that the gradient is only one line
                        grad[i] -= 0.5f * dv0;
                        grad[j] -= 0.5f * dv1;
                }

                return grad;
        }

        void step(const std::vector <glm::vec2> &grad, float dt) {
                for (int i = 0; i < curve.size(); i++) {
                        assert(!std::isnan(grad[i].x) && !std::isnan(grad[i].y));
                        curve[i] += dt * grad[i];
                }

                std::tie(min, max) = bounds(curve);
        }
};

int main()
{
        GLFWwindow *window = glfw_init();
        if (!window) {
                fprintf(stderr, "Failed to initialize GLFW\n");
                return -1;
        }

        std::vector <CurveRasterizer> rasterizers;

        {
                // TODO: generating random closed curves (non intersecting)
                Curve curve;

                float radius = 0.5;
                for (float theta = 0.0f; theta < 2.0f * M_PI; theta += 0.01f) {
                        float r = 2 * radius;
                        // for (int n = 0; n < 5; n++)
                        //         r += 0.4 * pow(0.5f, n) * sin(pow(2.0f, n) * theta);
                        // r = std::max(0.2f * radius, r);
                        // 
                        curve.push_back(glm::vec2 { r * cos(theta), r * sin(theta) });
                }

                printf("Curve elements size: %lu\n", curve.size());
                rasterizers.emplace_back(curve);
        }

        {
                Curve curve;

                float radius = 0.5;
                for (float theta = 0.0f; theta < 2.0f * M_PI; theta += 0.1f)
                        curve.push_back(glm::vec2 { radius * cos(theta), radius * sin(theta) });

                printf("Curve elements size: %lu\n", curve.size());
                rasterizers.emplace_back(curve);
        }

        CurveRasterizer &target = rasterizers[0];
        CurveRasterizer &source = rasterizers[1];
        target.rasterize();

        Image <SingleChannel> annotations { WINDOW_WIDTH, WINDOW_HEIGHT };
        Image <ColorChannel> final { WINDOW_WIDTH, WINDOW_HEIGHT };

        while (!glfwWindowShouldClose(window)) {
                // Render here
                glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                // Compute gradients and update
                // TODO: decay
                source.clear();
                annotations.clear();

                source.rasterize();
                auto derivatives = source.grad(target.interior, 100);
                // laplacian_diffuse(derivatives);

                std::vector <glm::vec2> ann_points;
                std::vector <glm::vec2> ann_derivatives;

                for (int i = 0; i < source.curve.size(); i++) {
                        ann_points.push_back(source.curve[i]);
                        ann_derivatives.push_back(derivatives[i]);
                }

                quiver(annotations, ann_points, ann_derivatives);
                source.step(derivatives, 0.01f);

                // Composite and draw
                composite(final, {
                        &rasterizers[0].boundary,
                        &rasterizers[0].interior,
                        &rasterizers[1].boundary,
                        &rasterizers[1].interior,
                        &annotations
                }, {
                        { 1.0f, 0.7f, 0.0f },
                        { 1.0f, 0.7f, 0.0f },
                        { 0.2f, 0.3f, 0.7f },
                        { 0.2f, 0.3f, 0.7f },
                        { 1.0f, 0.2f, 0.2f }
                });

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
