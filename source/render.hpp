#pragma once

// Standard headers
#include <vector>

// Local headers
#include "image.hpp"

// Compositing images
void composite(Image <ColorChannel> &dst,
               const std::vector <const Image <SingleChannel> *> &srcs,
               const std::vector <glm::vec3> &colors,
               const std::vector <float> &alpha_m)
{
        assert(srcs.size() == colors.size());
        
        // Use src images as the alpha (and use a weighted average)
        #pragma omp parallel for
        for (int i = 0; i < dst.width * dst.height; i++) {
                int x = i % dst.width;
                int y = i / dst.width;

                float u = x / float(dst.width);
                float v = y / float(dst.height);

                glm::vec3 color = dst(x, y);
                for (int i = 0; i < srcs.size(); i++) {
                        int x_ = u * srcs[i]->width;
                        int y_ = v * srcs[i]->height;

                        float alpha = (*srcs[i])(x_, y_) * alpha_m[i];
                        if (alpha < 1e-6f)
                                continue;

                        color = alpha * colors[i] + (1.0f - alpha) * color;
                }

                dst(x, y) = color;
        }
}

void composite(Image <ColorChannel> &dst,
               const Image <ColorChannel> &base,
               const std::vector <const Image <SingleChannel> *> &srcs,
               const std::vector <glm::vec3> &colors,
               const std::vector <float> &alpha_m)
{
        assert(srcs.size() == colors.size());
        assert(dst.width == base.width && dst.height == base.height);

        // Use src images as the alpha (and use a weighted average)
        #pragma omp parallel for
        for (int i = 0; i < dst.width * dst.height; i++) {
                int x = i % dst.width;
                int y = i / dst.width;

                float u = x / float(dst.width);
                float v = y / float(dst.height);

                glm::vec3 color = base(x, y);
                for (int i = 0; i < srcs.size(); i++) {
                        int x_ = u * srcs[i]->width;
                        int y_ = v * srcs[i]->height;

                        float alpha = (*srcs[i])(x_, y_) * alpha_m[i];
                        if (alpha < 1e-6f)
                                continue;

                        color = alpha * colors[i] + (1.0f - alpha) * color;
                }

                dst(x, y) = color;
        }
}

void convert(const Image <ColorChannel> &src, uint8_t *data)
{
        #pragma omp parallel for
        for (int y = 0; y < src.height; y++) {
                for (int x = 0; x < src.width; x++) {
                        int i = (y * src.width + x) * 4;
                        int y_ = src.height - y - 1;
                        data[i + 0] = src(x, y_).r * 255.0f;
                        data[i + 1] = src(x, y_).g * 255.0f;
                        data[i + 2] = src(x, y_).b * 255.0f;
                        data[i + 3] = 255;
                }
        }
}
