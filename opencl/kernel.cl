__kernel void multMatrices(__global int* left, __global int* right, __global int* result, const int len, const int step) {
    const int idx = get_global_id(0);
    int start = step * idx;
    int end = step * (idx + 1);
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < len; j++)
        {
            for (int k = 0; k < len; k++)
            {
                result[i * len + j] += left[i * len + k] * right[k * len + j];
            }
        }
    }
}
