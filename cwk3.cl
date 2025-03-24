// Kernel for the heat equation.
__kernel void heat(__global const float *in, __global float *out, int N) {
    // Obtain the 2D indices from the NDRange.
    int i = get_global_id(0);
    int j = get_global_id(1);
    int idx = i * N + j;
    
    // If the cell is on the boundary, set output to zero.
    if (i == 0 || j == 0 || i == N - 1 || j == N - 1) {
        out[idx] = 0.0f;
    } else {
        // Compute the average of the four neighboring cells (left, right, top, bottom).
        int idx_left = idx - 1;
        int idx_right = idx + 1;
        int idx_top = idx - N;
        int idx_bottom = idx + N;
        out[idx] = 0.25f * (in[idx_left] + in[idx_right] + in[idx_top] + in[idx_bottom]);
    }
}
