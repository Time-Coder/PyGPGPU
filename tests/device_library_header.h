/* A device-side function the user ships, the way a CUDA header declares one.

   The test registers it with `device_function`, so the generated source includes this
   header and calls the symbol it declares.
*/
__device__ float test_device_rcp(float x)
{
    return 1.0f / x;
}
