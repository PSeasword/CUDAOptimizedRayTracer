# Changes

* Passes origin and light by value as opposed to allocating host and device memory, transferring between the two
* Then, origin and light are passed by reference within the device
