/* A device function of your own, the way a CUDA header declares one.

   The example registers it with `device_function`, so the generated source includes this
   header and calls the symbol it declares.
*/
__device__ float my_fast_rcp(float x)
{
    return 1.0f / x;
}
