# Changes

Made changes to main.cu and files found in srcVec3f:
* Adjusted code formatting for improved readability
* Improved indentation and white space consistency
* Removed the use of auto for simple data type
* Added a significant number of comments to describe what the code actually does
* Divided main.cu into separate files
* Changed to CPU timer (around 20 microseconds difference from cudaEvent implementation) for kernel
* Changed CPU timer type for writing to file
* Added timers to different places in the code to allow for more profiling
* Added more spheres to the scene
* Changed reflections to also take the background color into account
* Merged Vec3f.cuh and Vec3f.cu
