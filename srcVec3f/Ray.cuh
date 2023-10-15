#pragma once

#include "Vec3f.cuh"

// Ray to be traced through the scene
struct Ray {
  Vec3f origin;
  Vec3f dir;

  __host__ __device__ Ray(const Vec3f &o, const Vec3f &d) : origin(o), dir(d) {}
  __host__ __device__ Vec3f at(float t) const;
  __host__ __device__ float has_intersection(const Sphere &sphere) const;
};

// Returns a point along the ray at distance t
__host__ __device__ Vec3f Ray::at(float t) const {
  return origin + (t * dir);
}

// Return intersection point of ray on sphere if it exists
__host__ __device__ float Ray::has_intersection(const Sphere& sphere) const {
  // Coefficients for quadratic equation of the intersection of ray and sphere
  float a = dot(dir, dir);
  float b = dot((2.0f * (dir)), (origin - sphere.center));
  float c = dot((origin - sphere.center), (origin - sphere.center)) - pow(sphere.radius, 2);

  // Discriminant of the quadratic equation
  float d = b*b - 4 * (a * c);

  // No real intersection
  if (d < 0) {
    return -1.0;
  }

  // The two possible intersection points
  float t0 = ((-b - std::sqrt(d)) / (2*a));
  float t1 = ((-b + std::sqrt(d)) / (2*a));

  // Only one intersection point
  if (d == 0) {
    return t0;
  }

  // Both intersections are behind the ray (ray moves opposite direction from potential intersection with sphere)
  if (t0 < 0 && t1 < 0) {
    return -1;
  }

  // One intersection point is in front of the ray
  if (t0 > 0 && t1 < 0) {
    return t0;
  }

  // One intersection point is in front of the ray
  if (t0 < 0 && t1 > 0) {
    return t1;
  }

  // Return the closest intersection point as both are in front of the ray
  return t0 < t1 ? t0 : t1;
}
