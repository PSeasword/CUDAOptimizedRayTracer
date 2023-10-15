#pragma once

#include "MathFloat3.cuh"

// Ray to be traced through the scene
struct Ray {
  float3 origin;
  float3 dir;

  __host__ __device__ Ray(const float3 &o, const float3 &d) : origin(o), dir(d) {}
  __host__ __device__ float3 at(const float &t) const;
  __host__ __device__ float has_intersection(const Sphere &sphere) const;
};

// Returns a point along the ray at distance t
__host__ __device__ float3 Ray::at(const float &t) const {
  return origin + (t * dir);
}

// Return intersection point of ray on sphere if it exists
__host__ __device__ float Ray::has_intersection(const Sphere &sphere) const {
  const float3 sphere_to_origin = origin - sphere.center;

  // Coefficients for quadratic equation of the intersection of ray and sphere
  const float a = dot_float3(dir, dir);
  const float b = dot_float3((2.0f * (dir)), sphere_to_origin);
  const float c = dot_float3(sphere_to_origin, sphere_to_origin) - sphere.radius * sphere.radius;

  // Discriminant of the quadratic equation
  const float d = b * b - 4 * (a * c);

  // No real intersection
  if (d < 0) {
    return -1.0;
  }

  // The two possible intersection points
  const float sqrtVal = sqrtf(d);
  const float t0 = ((-b - sqrtVal) / (2 * a));
  const float t1 = ((-b + sqrtVal) / (2 * a));

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
