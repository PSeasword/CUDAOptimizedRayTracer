#pragma once

#include <ostream>

class Vec3f {
  public:
    __host__ __device__ Vec3f(float x = 0, float y = 0, float z = 0) : mx(x), my(y), mz(z) {}

    __host__ __device__ Vec3f operator+(const Vec3f& a) const;
    __host__ __device__ Vec3f operator-(const Vec3f& a) const;
    __host__ __device__ Vec3f operator/(float a) const;
    __host__ __device__ Vec3f operator&(const Vec3f& a) const; // Element wise multiplication

    __host__ __device__ float length() const;
    __host__ __device__ Vec3f normalize() const;
    __host__ __device__ Vec3f scale(float factor) const;
    __host__ __device__ Vec3f cap(float max) const; // Sets the components of the vector to be within a maximum value

    __host__ __device__ float x() const;
    __host__ __device__ float y() const;
    __host__ __device__ float z() const;

  private:
    float mx; // Vector x component
    float my; // Vector y component
    float mz; // Vector z component
};

__host__ __device__ Vec3f operator*(float a, const Vec3f& b);
__host__ __device__ Vec3f operator*(const Vec3f& b, float a);

__host__ __device__ Vec3f cross(const Vec3f& a, const Vec3f& b);
// __host__ __device__ std::ostream& operator<<(std::ostream& os, const Vec3f& a);
__host__ __device__ float dot(const Vec3f& a, const Vec3f& b);
