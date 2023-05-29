#pragma once

// Standard headers
#include <map>
#include <vector>
#include <iostream>
#include <random>

// GLM headers
#include <glm/glm.hpp>

// Local headers
#include "raster.hpp"

// Optimizers
struct Optimizer {
        using UpscaleInfo = std::map <int, std::pair <int32_t, float>>;

        virtual float step(Curve &curve, const std::vector <glm::vec2> &grad) = 0;
        virtual void upscale(const Curve &curve, const UpscaleInfo &info) {}
};

struct SGD : Optimizer {
        float alpha;

        SGD(float alpha_) : alpha { alpha_ } {}

        float step(Curve &curve, const std::vector <glm::vec2> &grad) override {
                float grad_avg = 0.0f;
                for (int i = 0; i < curve.size(); i++) {
                        curve[i] += alpha * grad[i];
                        grad_avg += glm::length(grad[i]);
                }

                return grad_avg / curve.size();
        }
};

struct Momentum : Optimizer {
        std::vector <glm::vec2> v;

        float alpha;
        float eta;

        Momentum(float alpha_, float eta_ = 0.9)
                : alpha { alpha_ }, eta { eta_ } {}

        float step(Curve &curve, const std::vector <glm::vec2> &grad) override {
                if (v.size() != curve.size()) {
                        printf("Initializing momentum\n");
                        v.resize(curve.size());

                        // Clear the momentum
                        for (int i = 0; i < curve.size(); i++)
                                v[i] = glm::vec2 { 0.0f };
                }

                float grad_avg = 0.0f;
                for (int i = 0; i < curve.size(); i++) {
                        assert(!glm::any(glm::isnan(grad[i])));
                        v[i] = eta * v[i] + alpha * grad[i];
                        curve[i] += v[i];
                        grad_avg += glm::length(grad[i]);
                }

                return grad_avg / curve.size();
        }

        void upscale(const Curve &curve, const UpscaleInfo &new_indices) override {
                printf("Upscaling momentum to %d\n", curve.size());
                std::vector <glm::vec2> new_v(curve.size());
                for (int i = 0; i < curve.size(); i++) {
                        if (new_indices.count(i) == 0) {
                                new_v[i] = v[i];
                        } else {
                                auto pr = new_indices.at(i);
                                int j = pr.first;
                                int nj = (j + 1) % v.size();

                                new_v[i] = pr.second * v[j] + (1.0f - pr.second) * v[nj];
                        }
                }

                v = std::move(new_v);
        }
};

struct Adam : Optimizer {
        std::vector <glm::vec2> m;
        std::vector <glm::vec2> v;

        float alpha;
        float beta1;
        float beta2;

        float t;

        Adam(float alpha_, float beta1_ = 0.9, float beta2_ = 0.999)
                : alpha { alpha_ }, beta1 { beta1_ }, beta2 { beta2_ }, t { 0.0f } {}

        float step(Curve &curve, const std::vector <glm::vec2> &grad) override {
                if (m.size() != curve.size()) {
                        printf("Initializing Adam\n");
                        m.resize(curve.size());
                        v.resize(curve.size());

                        // Clear the momentum
                        for (int i = 0; i < curve.size(); i++) {
                                m[i] = glm::vec2 { 0.0f };
                                v[i] = glm::vec2 { 0.0f };
                        }
                }

                t += 1.0f;
                float grad_avg = 0.0f;
                for (int i = 0; i < curve.size(); i++) {
                        assert(!glm::any(glm::isnan(grad[i])));
                        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
                        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

                        glm::vec2 m_hat = m[i] / (1.0f - pow(beta1, t));
                        glm::vec2 v_hat = v[i] / (1.0f - pow(beta2, t));

                        curve[i] += alpha * m_hat / (sqrt(v_hat) + 1e-8f);
                        grad_avg += glm::length(grad[i]);
                }

                return grad_avg / curve.size();
        }

        void upscale(const Curve &curve, const UpscaleInfo &new_indices) override {
                printf("Upscaling Adam to %d\n", curve.size());
                std::vector <glm::vec2> new_m(curve.size());
                std::vector <glm::vec2> new_v(curve.size());
                for (int i = 0; i < curve.size(); i++) {
                        if (new_indices.count(i) == 0) {
                                new_m[i] = m[i];
                                new_v[i] = v[i];
                        } else {
                                auto pr = new_indices.at(i);
                                int j = pr.first;
                                int nj = (j + 1) % v.size();

                                new_m[i] = pr.second * m[j] + (1.0f - pr.second) * m[nj];
                                new_v[i] = pr.second * v[j] + (1.0f - pr.second) * v[nj];
                        }
                }

                m = std::move(new_m);
                v = std::move(new_v);
        }
};

// Wrapper struct
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

        std::vector <glm::vec2> normal() {
                std::vector <glm::vec2> normal { curve.size(), glm::vec2 { 0.0f } };
                for (int i = 0; i < curve.size(); i++) {
                        int j = (i + 1) % curve.size();

                        glm::vec2 p0 = curve[i];
                        glm::vec2 p1 = curve[j];

                        glm::vec2 d = glm::normalize(p1 - p0);
                        glm::vec2 n = glm::normalize(glm::vec2(-d.y, d.x));

                        normal[i] = n;
                }

                return normal;
        }

        float length() {
                float length = 0.0f;
                for (int i = 0; i < curve.size(); i++) {
                        int j = (i + 1) % curve.size();

                        glm::vec2 p0 = curve[i];
                        glm::vec2 p1 = curve[j];

                        length += glm::length(p1 - p0);
                }

                return length;
        }

        std::vector <glm::vec2> grad(const Image <SingleChannel> &target, int N) {
                std::mt19937 rng { std::random_device {}() };
                std::uniform_real_distribution <float> dist { 0.0f, 1.0f };

                std::vector <glm::vec2> grad { curve.size(), glm::vec2 { 0.0f } };

                // TODO: double buffer to avoid race conditions
                std::vector <glm::vec2> grads0 { curve.size() };
                std::vector <glm::vec2> grads1 { curve.size() };

                #pragma omp parallel for
                for (int i = 0; i < curve.size(); i++) {
                        int j = (i + 1) % curve.size();

                        glm::vec2 p0 = curve[i];
                        glm::vec2 p1 = curve[j];

                        glm::vec2 dv0 { 0.0f };
                        glm::vec2 dv1 { 0.0f };

                        for (int k = 0; k < N; k++) {
                                float t = dist(rng);
                                auto [sdv0, sdv1] = grad_segment(target, interior, { p0, p1 }, t);
                                
                                dv0 += sdv0;
                                dv1 += sdv1;
                        }
                        
                        dv0 /= N;
                        dv1 /= N;

                        // Account for the fact that the gradient is only one line
                        grads0[i] = 0.5f * dv0;
                        grads1[j] = 0.5f * dv1;
                }

                #pragma omp parallel for
                for (int i = 0; i < curve.size(); i++)
                        grad[i] = grads0[i] + grads1[i];

                return grad;
        }

        Optimizer::UpscaleInfo adaptive_subdivide(const std::vector <glm::vec2> &grad) {
                float redist = 0.8 * length() / curve.size();

                // Subdivide based on redistributing length
                std::vector <glm::vec2> new_curve;
                Optimizer::UpscaleInfo new_indices;
                for (int i = 0; i < curve.size(); i++) {
                        int j = (i + 1) % curve.size();

                        glm::vec2 p0 = curve[i];
                        glm::vec2 p1 = curve[j];

                        float d = glm::length(p1 - p0);
                        int n = std::max(1, (int)std::round(d/redist));

                        for (int k = 0; k < n; k++) {
                                float t = (float) k / (float) n;
                                glm::vec2 p = (1.0f - t) * p0 + t * p1;

                                new_indices[new_curve.size()] = { i, t };
                                new_curve.push_back(p);
                        }
                }

                curve = new_curve;
                update();

                return new_indices;
        }

        void update() {
                std::tie(min, max) = bounds(curve);
        }
};
