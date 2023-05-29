#pragma once

// Standard headers
#include <optional>

// GLM headers
#include <glm/glm.hpp>

// Local headers
#include "image.hpp"
#include "raster.hpp"

// Intersection over union
float iou(const Image <SingleChannel> &a, const Image <SingleChannel> &b)
{
        int width = std::min(a.width, b.width);
        int height = std::min(a.height, b.height);

        float intersection = 0.0f;
        float union_ = 0.0f;

        for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                        float u = x / float(width);
                        float v = y / float(height);

                        int ax = int(u * a.width);
                        int ay = int(v * a.height);

                        int bx = int(u * b.width);
                        int by = int(v * b.height);

                        intersection += a(ax, ay) * b(bx, by);
                        union_ += std::max(a(ax, ay), b(bx, by));
                }
        }

        intersection /= width * height;
        union_ /= width * height;

        return intersection / union_;
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

        // glm::vec2 screen_p = viewport.to_screen(p);
        glm::vec2 screen_uv = (p + glm::vec2 { viewport.xrange, viewport.yrange }) / (2.0f * glm::vec2 { viewport.xrange, viewport.yrange });
        glm::vec2 screen_p = screen_uv * glm::vec2 { target.width, target.height };

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
                        glm::vec2 uv = q / glm::vec2 { target.width, target.height };
                        glm::vec2 pt = viewport.to_cartesian(uv);

                        // Skip if on the line
                        if (std::fabs(glm::dot(pt - line.v0, line.v1 - line.v0)) < 1e-6f)
                                continue;

                        float w = glm::length(pt - p);

                        float at = target(uv.x * target.width, uv.y * target.height);
                        float ai = interior(uv.x * interior.width, uv.y * interior.height);
                        float delta = std::abs(ai - at);
                        
                        dp += delta * n * w * glm::sign(glm::dot(pt - p, n));
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

// Intersection point of two line segments
inline float cross(const glm::vec2 &a, const glm::vec2 &b)
{
        return a.x * b.y - a.y * b.x;
}

// Intersection time of ray and circle
float intersect(const glm::vec2 &o, const glm::vec2 &d, const glm::vec2 &c, float r)
{
        // Return -1.0f if no intersection or out of bounds (e.g. t > 1.0f)
        float a = glm::dot(d, d);
        float b = 2.0f * glm::dot(d, o - c);
        float c_ = glm::dot(o - c, o - c) - r * r;

        float delta = b * b - 4.0f * a * c_;

        if (delta < 0.0f)
                return -1.0f;

        float t0 = (-b - std::sqrt(delta)) / (2.0f * a);
        float t1 = (-b + std::sqrt(delta)) / (2.0f * a);

        float t = t0 > 0.0f ? t0 : t1;

        if (t < 0.0f || t > 1.0f)
                return -1.0f;

        return t;
}

std::optional <glm::vec2> intersection_point(const LineSegment &a, const LineSegment &b)
{
        glm::vec2 p = a.v0;
        glm::vec2 r = a.v1 - a.v0;
        glm::vec2 q = b.v0;
        glm::vec2 s = b.v1 - b.v0;

        float t = cross(q - p, s) / cross(r, s);
        float u = cross(q - p, r) / cross(r, s);

        if (t < 0.0f || t > 1.0f || u < 0.0f || u > 1.0f)
                return std::nullopt;

        return p + t * r;
}

// Gradient regularizers
void laplacian_diffuse(std::vector <glm::vec2> &grad, int power)
{
        auto copy = grad;

        for (int i = 0; i < power; i++) {
                for (int i = 0; i < grad.size(); i++) {
                        int pi = (i - 1 + grad.size()) % grad.size();
                        int ni = (i + 1) % grad.size();

                        grad[i] = (copy[pi] + 2.0f * copy[i] + copy[ni]) / 4.0f;
                }

                if (i + 1 < power)
                        std::swap(grad, copy);
        }
}
