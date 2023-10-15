#pragma once

#include "Vec3f.cuh"

// A 3D sphere rendered within the scene
struct Sphere {
  float radius;
  Vec3f center;
  Color color;

  __host__ __device__ Sphere() {}
  __host__ __device__ Sphere(float r, const Vec3f &c, const Vec3f &col) : radius(r), center(c), color(col) {}
  __host__ __device__ Vec3f get_normal_at(const Vec3f &at) const;
};

// Get normal vector at point on surface
__host__ __device__ Vec3f Sphere::get_normal_at(const Vec3f& at) const {
  return Vec3f(at - center).normalize();
}
