# Changes

* Added break to the two for-loops that searches for spherea that creates a shadow
* Removed the array of intersections and replaced it with only the closest intersection
* Use ++i instead of i++ (only one change)
* Removed redundant if-statements in get_closest_intersection()
* Added const to multiple variables
* Combined ambient, diffuse, and specular in get_color_at
* Removed intermediate step of writing to mem_buffer when writing to file
*Removed static cast of color as it is already done when writing the frame buffer to file
* Optimized Ray::has_intersection
* Other small changes
