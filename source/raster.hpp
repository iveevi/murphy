#pragma once

// Standard headers
#include <algorithm>
#include <map>
#include <set>
#include <vector>

// GLM headers
#include <glm/glm.hpp>
#include <glm/gtx/closest_point.hpp>
#include <glm/gtx/rotate_vector.hpp>

// Local headers
#include "image.hpp"

inline int WINDOW_WIDTH = 750;
inline int WINDOW_HEIGHT = 750;

// TODO: profiler
// Render viewport configuration
struct {
        // TODO: make specific to images...
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
} inline viewport;

// Rasterizing lines
struct LineSegment {
        glm::vec2 v0;
        glm::vec2 v1;
};

// Closest distance from a point to a line segment
inline float distance(const LineSegment &line, const glm::vec2 &point)
{
        return glm::length(glm::closestPointOnLine(point, line.v0, line.v1) - point);
}

inline void rasterize_line(Image <SingleChannel> &img, const LineSegment &line)
{
        float thickness = 2.0f * viewport.epsilon;
        glm::vec2 line_min = viewport.to_uv(glm::min(line.v0, line.v1) - thickness);
        glm::vec2 line_max = viewport.to_uv(glm::max(line.v0, line.v1) + thickness);

        int min_y = std::max(0, int(line_min.y * img.height));
        int max_y = std::min(img.height - 1, int(line_max.y * img.height));

        int min_x = std::max(0, int(line_min.x * img.width));
        int max_x = std::min(img.width - 1, int(line_max.x * img.width));

        // TODO: line search from min y to max y (including thickness)
        #pragma omp parallel for
        for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                        glm::vec2 uv = glm::vec2 { x, y } / glm::vec2 { img.width, img.height };
                        glm::vec2 point = glm::vec2 { viewport.xrange, viewport.yrange } * (uv - 0.5f) * 2.0f;
                        float dist = distance(line, point);
                        float c = 1.0f - glm::smoothstep(0.0f, thickness, dist);
                        img(x, y) = std::max(img(x, y), c);
                }
        }
}

// Pooled line rasterization
inline void rasterize_lines(Image <SingleChannel> &img, const std::vector <LineSegment> &lines)
{
        // Generate list of all pixels that are within the line segments
        // as well as the line segments associated with them
        struct ScanLine {
                std::set <int32_t> xs;
                std::set <int32_t> line_ids;
        };

        std::map <int32_t, ScanLine> line_verticals;

        for (int i = 0; i < lines.size(); i++) {
                const LineSegment &l = lines[i];
                float thickness = 2.0f * viewport.epsilon;
                glm::vec2 line_min = viewport.to_uv(glm::min(l.v0, l.v1) - thickness);
                glm::vec2 line_max = viewport.to_uv(glm::max(l.v0, l.v1) + thickness);

                int min_y = std::max(0, int(line_min.y * img.height));
                int max_y = std::min(img.height - 1, int(line_max.y * img.height));

                for (int y = min_y; y <= max_y; y++) {
                        if (line_verticals.find(y) == line_verticals.end())
                                line_verticals[y] = ScanLine();

                        ScanLine &scanline = line_verticals[y];
                        line_verticals[y].line_ids.insert(i);

                        int min_x = std::max(0, int(line_min.x * img.width));
                        int max_x = std::min(img.width - 1, int(line_max.x * img.width));

                        for (int x = min_x; x <= max_x; x++)
                                scanline.xs.insert(x);
                }
        }

        // Linearize the scanlines
        struct LinearScanLine {
                std::vector <int32_t> xs;
                std::vector <int32_t> line_ids;
        };

        std::vector <std::pair <int32_t, LinearScanLine>> line_verticals_linear;
        for (auto &kv : line_verticals) {
                int y = kv.first;
                ScanLine &scanline = kv.second;

                LinearScanLine linear_scanline;
                linear_scanline.xs.reserve(scanline.xs.size());
                linear_scanline.line_ids.reserve(scanline.line_ids.size());

                for (int x : scanline.xs)
                        linear_scanline.xs.push_back(x);

                for (int line_id : scanline.line_ids)
                        linear_scanline.line_ids.push_back(line_id);

                line_verticals_linear.push_back(std::make_pair(y, linear_scanline));
        }

        // Rasterize each scanline
        #pragma omp parallel for
        for (int i = 0; i < line_verticals_linear.size(); i++) {
                int y = line_verticals_linear[i].first;
                LinearScanLine &linear_scanline = line_verticals_linear[i].second;

                for (int j = 0; j < linear_scanline.xs.size(); j++) {
                        int x = linear_scanline.xs[j];
                        glm::vec2 uv = glm::vec2 { x, y } / glm::vec2 { img.width, img.height };
                        glm::vec2 point = glm::vec2 { viewport.xrange, viewport.yrange } * (uv - 0.5f) * 2.0f;

                        float min_dist = std::numeric_limits <float>::infinity();
                        for (int line_id : linear_scanline.line_ids) {
                                const LineSegment &l = lines[line_id];
                                float dist = distance(l, point);
                                min_dist = std::min(min_dist, dist);
                        }

                        float c = 1.0f - glm::smoothstep(0.0f, 2.0f * viewport.epsilon, min_dist);
                        img(x, y) = std::max(img(x, y), c);
                }
        }
}

inline void rasterize_arrow(Image <SingleChannel> &img, const glm::vec2 &origin, const glm::vec2 &direction)
{
        glm::vec2 v0 = origin;
        glm::vec2 v1 = origin + direction;

        glm::vec2 left = -glm::rotate(direction, glm::radians(30.0f))/10.0f;
        glm::vec2 right = -glm::rotate(direction, glm::radians(-30.0f))/10.0f;

        rasterize_line(img, { v0, v1 });
        rasterize_line(img, { v1, v1 + left });
        rasterize_line(img, { v1, v1 + right });
}

inline void preraster_arrow(std::vector <LineSegment> &items, const glm::vec2 &origin, const glm::vec2 &direction)
{
        glm::vec2 v0 = origin;
        glm::vec2 v1 = origin + direction;

        glm::vec2 left = -glm::rotate(direction, glm::radians(30.0f))/10.0f;
        glm::vec2 right = -glm::rotate(direction, glm::radians(-30.0f))/10.0f;

        items.push_back({ v0, v1 });
        items.push_back({ v1, v1 + left });
        items.push_back({ v1, v1 + right });
}

inline void quiver(Image <SingleChannel> &img, const std::vector <glm::vec2> &points, const std::vector <glm::vec2> &directions)
{
        for (int i = 0; i < points.size(); i++)
                rasterize_arrow(img, points[i], directions[i]);
}

inline void preraster_quiver(std::vector <LineSegment> &items, const std::vector <glm::vec2> &points, const std::vector <glm::vec2> &directions)
{
        for (int i = 0; i < points.size(); i++)
                preraster_arrow(items, points[i], directions[i]);
}

using Curve = std::vector <glm::vec2>;

inline void rasterize_boundary(Image <SingleChannel> &img, const Curve &curve)
{
        for (int i = 0; i < curve.size(); i++) {
                int j = (i + 1) % curve.size();
                rasterize_line(img, { curve[i], curve[j] });
        }

        // std::vector <LineSegment> lines;
        // for (int i = 0; i < curve.size(); i++) {
        //         int j = (i + 1) % curve.size();
        //         lines.push_back({ curve[i], curve[j] });
        // }
        //
        // rasterize_lines(img, lines);
}

inline std::pair <glm::vec2, glm::vec2> bounds(const Curve &curve)
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
inline void rasterize_interior(Image <SingleChannel> &dst, const Curve &curve, const glm::vec2 &min, const glm::vec2 &max)
{
        // Use analytical horizontal line scan
        float thickness = 2.0f * viewport.epsilon;
        glm::vec2 min_uv = viewport.to_uv(min - thickness);
        glm::vec2 max_uv = viewport.to_uv(max + thickness);

        int min_x = std::max(0, int(min_uv.x * dst.width));
        int min_y = std::max(0, int(min_uv.y * dst.height));
        int max_x = std::min(dst.width, int(max_uv.x * dst.width));
        int max_y = std::min(dst.height, int(max_uv.y * dst.height));

        #pragma omp parallel for
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
