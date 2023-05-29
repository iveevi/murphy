#pragma once

// GLM headers
#include <glm/glm.hpp>

// Graphics headers
#include "gl.hpp"

struct SingleChannel {
        using type = float;
        static constexpr int gl = GL_RED;
        static constexpr int channels = 1;
        static constexpr float clear = 0.0f;
        static constexpr float luminance(float val) {
                return val;
        }
};

struct ColorChannel {
        using type = glm::vec3;
        static constexpr int gl = GL_RGB;
        static constexpr int channels = 3;
        static constexpr glm::vec3 clear = glm::vec3 { 0.0f };
        static constexpr float luminance(const glm::vec3 &val) {
                return glm::dot(val, glm::vec3 { 0.2126f, 0.7152f, 0.0722f });
        }
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

        void copy(const Image &other) {
                // If same dimensions, plain copy, otherwise use UV sampling
                if (width == other.width && height == other.height) {
                        std::copy(other.data, other.data + width * height * Channel::channels, data);
                } else {
                        #pragma omp parallel for
                        for (int y = 0; y < height; y++) {
                                for (int x = 0; x < width; x++) {
                                        float u = float(x) / width;
                                        float v = float(y) / height;

                                        int ox = u * other.width;
                                        int oy = v * other.height;

                                        data[y * width + x] = other.data[oy * other.width + ox];
                                }
                        }
                }
        }

        void resize(int w, int h) {
                Image <Channel> other(w, h);
                other.copy(*this);
                *this = std::move(other);
        }

        void clear(typename Channel::type val = typename Channel::type { 0 }) {
                for (int i = 0; i < width * height * Channel::channels; i++)
                        data[i] = val;
        }
};

// General blurring
// TODO: gaussian blur
template <typename Channel>
Image <Channel> smooth(const Image <Channel> &image, int K)
{
        Image <Channel> out(image.width, image.height);

        #pragma omp parallel for
        for (int y = 0; y < image.height; y++) {
                for (int x = 0; x < image.width; x++) {
                        typename Channel::type sum = typename Channel::type { 0 };
                        for (int dy = -K; dy <= K; dy++) {
                                for (int dx = -K; dx <= K; dx++) {
                                        int nx = x + dx;
                                        int ny = y + dy;
                                        if (nx < 0 || nx >= image.width || ny < 0 || ny >= image.height)
                                                continue;
                                        sum += image(nx, ny);
                                }
                        }

                        out(x, y) = sum / float((2 * K + 1) * (2 * K + 1));
                }
        }

        return out;
}

// Sobel edge detection
template <typename Channel>
Image <SingleChannel> sobel(const Image <Channel> &image)
{
        Image <SingleChannel> out(image.width, image.height);

        #pragma omp parallel for
        for (int y = 0; y < image.height; y++) {
                for (int x = 0; x < image.width; x++) {
                        float gx = 0.0f;
                        float gy = 0.0f;

                        for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                        int nx = x + dx;
                                        int ny = y + dy;
                                        if (nx < 0 || nx >= image.width || ny < 0 || ny >= image.height)
                                                continue;

                                        float coeff = (dx == 0 || dy == 0) ? 1.0f : 0.707f;
                                        float val = Channel::luminance(image(nx, ny));

                                        gx += coeff * val * dx;
                                        gy += coeff * val * dy;
                                }
                        }

                        out(x, y) = glm::length(glm::vec2 { gx, gy });
                }
        }

        return out;
}
