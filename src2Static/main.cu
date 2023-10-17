#include <iostream>
#include <iterator>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>

#include "../srcVec3f/Consts.cuh"
#include "../srcVec3f/cuda_util.cuh"
#include "../srcVec3f/Vec3f.cuh"
#include "../srcVec3f/Light.cuh"
#include "../srcVec3f/Sphere.cuh"
#include "../srcVec3f/RayOptimized.cuh"

// CPU Timer
auto start_CPU = std::chrono::high_resolution_clock::now();

void start_CPU_timer(){
    start_CPU = std::chrono::high_resolution_clock::now();
}

long stop_CPU_timer(const char* info){
    auto elapsed = std::chrono::high_resolution_clock::now() - start_CPU;
    long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << microseconds << " microseconds\t\t" << info << std::endl;
    return microseconds;
}

// Maximum of two floats
__device__ constexpr float f_max(float a, float b) {
  return a > b ? a : b;
}

// Find the closest intersecting sphere of a ray if it exists and set the closest intersection of the sphere if it exists
__device__ int get_closest_intersection(Sphere* spheres, const Ray& r, float* closest_intersection) {
  int hp = -1;
  *closest_intersection = 100.0;
  float current_intersection;
  
  for (int ii = 0; ii < OBJ_COUNT; ++ii) {
    current_intersection = r.has_intersection(spheres[ii]);

    // Intersection exists and is in front of the ray and the current intersection is closer than the previous one
    if (current_intersection >= 0.0 && current_intersection < *closest_intersection) {
      *closest_intersection = current_intersection;
      hp = ii;
    }
  }

  return hp;
}

// Calculate the color to display at the intersection between ray and sphere
__device__ Color get_color_at(const Ray &r, float intersection, Light* light, const Sphere &sphere, Sphere* spheres, Vec3f* origin) {
  const float surface_offset = 0.001;
  const float shadow_threshold = 0.000001;
  float shadow = 1; // Initialize shadow to full brightness

  Vec3f normal = sphere.get_normal_at(r.at(intersection));

  // Normalized vector from intersection point to camera
  Vec3f to_camera(*origin - r.at(intersection));
  to_camera = to_camera.normalize();

  // Normalized vector from intersection point to light source
  Vec3f light_ray(light->get_position() - r.at(intersection));
  light_ray = light_ray.normalize();

  // Normalized vector of the reflected ray when hitting a sphere
  Vec3f reflection_ray = (-1 * light_ray) - 2 * dot((-1 * light_ray), normal) * normal;
  reflection_ray = reflection_ray.normalize();

  // Reflection ray from intersection point
  Ray rr(r.at(intersection) + surface_offset * normal, reflection_ray); // Offset ray from surface so that it does not hit the same surface it just reflected away from
  float reflect_closest_intersection;
  int hp = get_closest_intersection(spheres, rr, &reflect_closest_intersection); // What the reflection ray hits
  float reflect_shadow = 1;
  Color reflect_color = Vec3f(BGD_R, BGD_G, BGD_B) / 255; // Set color in reflection to be background color by default

  // Reflection ray hit a sphere
  if (hp != -1) {
    // Ray from intersection point on the sphere that is reflected towards the light source
    Ray rs(rr.at(reflect_closest_intersection) + surface_offset * spheres[hp].get_normal_at(rr.at(reflect_closest_intersection)), light->get_position() - rr.at(reflect_closest_intersection) + surface_offset * spheres[hp].get_normal_at(rr.at(reflect_closest_intersection)));

    // Check if ray from intersection point on the sphere that is reflected towards the light source hits any sphere that creates a shadow
    for (int i = 0; i < OBJ_COUNT; ++i) {
      // There is a a sphere creating a shadow on the reflected sphere
      if (rs.has_intersection(spheres[i]) > shadow_threshold) {
        reflect_shadow = 0.35;
        break;
      }
    }

    reflect_color = reflect_shadow * spheres[hp].color;
  }

  // Calculate ambient, diffuse, and specular components of the light
  Vec3f ambient_diffuse_specular;
  ambient_diffuse_specular = ambient_diffuse_specular + light->get_ambient() * light->get_color(); // Ambient
  ambient_diffuse_specular = ambient_diffuse_specular + (light->get_diffuse() * f_max(dot(light_ray, normal), 0.0f)) * light->get_color(); // Diffuse
  ambient_diffuse_specular = ambient_diffuse_specular + light->get_specular() * pow(f_max(dot(reflection_ray, to_camera), 0.0f), 32) * light->get_color(); // Specular
  
  // Ray from interesection point on original sphere towards the light source
  Ray shadow_ray(r.at(intersection) + (surface_offset * normal), light->get_position() - (r.at(intersection) + surface_offset * normal));

  // Check if ray from intersection point on original sphere towards the light source hits any sphere that creates a shadow
  for (int i = 0; i < OBJ_COUNT; ++i) {
    // There is a sphere creating a shadow on the original sphere
    if (shadow_ray.has_intersection(spheres[i]) > shadow_threshold) {
      shadow = 0.35;
      break;
    }
  }
  
  // Final color before adding the shadow on the original sphere
  Vec3f all_light = (ambient_diffuse_specular).cap(1) & (0.55 * (sphere.color - reflect_color) + reflect_color).cap(1);
  
  return 255.999 * (shadow * all_light);
}

// Cast one ray per pixel
__global__ void cast_ray(Vec3f* fb, Sphere* spheres, Light* light, Vec3f* origin) {
  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  const int tid = (j*WIDTH) + i;

  // Outside rendered pixels
  if (i >= WIDTH || j >= HEIGHT) {
    return;
  }

  // Calculate the ray from the position of the camera toward the 3D scene through the current pixel on the 2D image plane
  const Vec3f ij(2 * (float((i) + 0.5) / (WIDTH - 1)) - 1, 1 - 2 * (float((j) + 0.5) / (HEIGHT - 1)), -1); // Direction vector for the ray
  const Vec3f dir(ij - *origin);
  Ray r(*origin, dir);

  float closest_intersection; // The closest intersection of the closest intersecting sphere
  const int hp = get_closest_intersection(spheres, r, &closest_intersection); // The closest intersecting sphere

  // Did not hit any spheres (background color)
  if (hp == -1) {
    fb[tid] = Vec3f(BGD_R, BGD_G, BGD_B);
  }
  // Did hit a sphere
  else {
    fb[tid] = get_color_at(r, closest_intersection, light, spheres[hp], spheres, origin);
  }
}

void initDevice(int& device_handle) {
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  printDeviceProps(devProp);

  cudaSetDevice(device_handle);
}

void run_kernel(const int pixels, Vec3f* fb, Sphere* spheres, Light* light, Vec3f* origin) {
  // Device
  Vec3f* fb_device = nullptr;
  Sphere* spheres_dv = nullptr;
  Light* light_dv = nullptr;
  Vec3f* origin_dv = nullptr;

  start_CPU_timer();

  // Device memory allocation
  checkErrorsCuda(cudaMalloc((void**) &fb_device, sizeof(Vec3f) * pixels));
  checkErrorsCuda(cudaMalloc((void**) &spheres_dv, sizeof(Sphere) * OBJ_COUNT));
  checkErrorsCuda(cudaMalloc((void**) &light_dv, sizeof(Light) * 1));
  checkErrorsCuda(cudaMalloc((void**) &origin_dv, sizeof(Vec3f) * 1));

  stop_CPU_timer("Device memory allocation");
  start_CPU_timer();

  // Host to device memory transfer
  checkErrorsCuda(cudaMemcpy((void*) fb_device, fb, sizeof(Vec3f) * pixels, cudaMemcpyHostToDevice));
  checkErrorsCuda(cudaMemcpy((void*) spheres_dv, spheres, sizeof(Sphere) * OBJ_COUNT, cudaMemcpyHostToDevice));
  checkErrorsCuda(cudaMemcpy((void*) light_dv, light, sizeof(Light) * 1, cudaMemcpyHostToDevice));
  checkErrorsCuda(cudaMemcpy((void*) origin_dv, origin, sizeof(Vec3f) * 1, cudaMemcpyHostToDevice));

  stop_CPU_timer("HtoD memory transfer");
  start_CPU_timer();

  // Launch kernel
  dim3 blocks(WIDTH / 16, HEIGHT / 16);
  cast_ray<<<blocks, dim3(16, 16)>>>(fb_device, spheres_dv, light_dv, origin_dv);

  cudaDeviceSynchronize();

  stop_CPU_timer("CUDA kernel");
  start_CPU_timer();

  // Device to host memory transfer
  checkErrorsCuda(cudaMemcpy(fb, fb_device, sizeof(Vec3f) * pixels, cudaMemcpyDeviceToHost));

  stop_CPU_timer("DtoH memory transfer");
  start_CPU_timer();

  // Free device memory
  checkErrorsCuda(cudaFree(fb_device));
  checkErrorsCuda(cudaFree(spheres_dv));
  checkErrorsCuda(cudaFree(light_dv));
  checkErrorsCuda(cudaFree(origin_dv));

  stop_CPU_timer("Freeing device memory");
}

int main(int argc, char *argv[]) {
  int write_to_file = true;
  
  if (argc == 2) {
    write_to_file = atoi(argv[1]);
  }

  std::ofstream file("img.ppm");

  const int pixels = WIDTH * HEIGHT;
  int device_handle = 0;

  int deviceCount = 0;
  checkErrorsCuda(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "initDevice(): No CUDA Device found." << std::endl;
    return EXIT_FAILURE;
  }

  initDevice(device_handle);
  
  std::cout << "===========================================" << std::endl;

  // Host memory allocation
  start_CPU_timer();

  // Frame buffer for all pixels
  Vec3f* frame_buffer = new Vec3f[pixels];

  // Create an array of spheres
  Sphere *spheres = new Sphere[OBJ_COUNT] {
    Sphere(1000, Vec3f(0, -1002, 0), Color(0.5, 0.5, 0.5)),
    Sphere(0.25, Vec3f(-1.5, -0.25, -4), Color(1.0, 0.0, 0.0)),
    Sphere(0.25, Vec3f(-1.0, -0.25, -4), Color(1.0, 0.5, 0.0)),
    Sphere(0.25, Vec3f(-0.5, -0.25, -4), Color(1.0, 1.0, 0.0)),
    Sphere(0.25, Vec3f(0, -0.25, -4), Color(0.0, 1.0, 0.0)),
    Sphere(0.25, Vec3f(0.5, -0.25, -4), Color(0.0, 1.0, 1.0)),
    Sphere(0.25, Vec3f(1.0, -0.25, -4), Color(0.0, 0.0, 1.0)),

    Sphere(0.25, Vec3f(1.5, -0.25, -4), Color(0.5, 0.0, 1.0)),
    Sphere(0.25, Vec3f(-1.25, 0.25, -3), Color(1.0, 0.0, 0.5)),
    Sphere(0.25, Vec3f(-0.75, 0.25, -3), Color(0.5, 0.0, 0.5)),
    Sphere(0.25, Vec3f(-0.25, 0.25, -3), Color(0.5, 0.5, 0.5)),
    Sphere(0.25, Vec3f(0.25, 0.25, -3), Color(1.0, 1.0, 0.5)),
    Sphere(0.25, Vec3f(0.75, 0.25, -3), Color(0.0, 1.0, 0.5)),

    Sphere(0.25, Vec3f(1.25, 0.25, -3), Color(0.0, 0.5, 1.0)),
    Sphere(0.25, Vec3f(-1.0, 0.75, -2), Color(1.0, 0.5, 0.0)),
    Sphere(0.25, Vec3f(-0.5, 0.75, -2), Color(0.0, 1.0, 1.0)),
    Sphere(0.25, Vec3f(0, 0.75, -2), Color(0.5, 0.0, 1.0)),
    Sphere(0.25, Vec3f(0.5, 0.75, -2), Color(0.0, 0.5, 0.0)),
    Sphere(0.25, Vec3f(1.0, 0.75, -2), Color(1.0, 1.0, 1.0)),
  };

  // Origin of the camera
  Vec3f *origin = new Vec3f(0, 0, 1);

  // Light source in the scene
  Light *light = new Light(Vec3f(1, 1, 1), Vec3f(1, 1, 1));
  // light->set_light(.2, .5, .5);
  light->set_light(.1, .7, .7);

  stop_CPU_timer("Host memory allocation");

  std::cout << ">> Starting kernel for " << WIDTH << "x" << HEIGHT << " image..." << std::endl;
  run_kernel(pixels, frame_buffer, spheres, light, origin);
  std::cout << ">> Finished kernel" << std::endl;

  std::cout << ">> Saving Image..." << std::endl;

  start_CPU_timer();
  
  if (write_to_file == 1) {
    // Write from the frame buffer to image file
    file << "P3" << "\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";

    for (std::size_t i = 0; i < pixels; ++i) {
      file << static_cast<int>(frame_buffer[i].x()) << " " << static_cast<int>(frame_buffer[i].y()) << " " << static_cast<int>(frame_buffer[i].z()) << "\n";
    }
  }

  stop_CPU_timer("Writing to file");

  start_CPU_timer();

  // Free host memory
  delete[] frame_buffer;
  delete origin;
  delete light;
  delete[] spheres;

  stop_CPU_timer("Freeing host memory");

  std::cout << "===========================================" << std::endl;

  return EXIT_SUCCESS;
}
