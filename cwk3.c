//
// Starting point for the OpenCL coursework for COMP/XJCO3221 Parallel Computation.
//
// Once compiled, execute with the size of the square grid as a command line argument, i.e.
//
// ./cwk3 16
//
// will generate a 16 by 16 grid. The C-code below will then display the initial grid,
// followed by the same grid again. You will need to implement OpenCL that applies the heat
// equation as per the instructions, so that the final grid is different.
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>

// For this coursework, the helper file has 2 routines in addition to simpleOpenContext_GPU() and compileKernelFromFile():
// getCmdLineArg()  :  Parses grid size N from command line argument, or fails with error message.
// fillGrid()       :  Fills the grid with random values, except boundary values which are always zero.
// Do not alter these routines, as they will be replaced with different versions for assessment.
#include "helper_cwk.h"


//
// Main.
//
int main( int argc, char **argv )
{
 
    //
    // Parse command line argument and check it is valid. Handled by a routine in the helper file.
    //
    int N;
    getCmdLineArg( argc, argv, &N );

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    // Allocate memory for the grid. For simplicity, this uses a one-dimensional array.
	float *hostGrid = (float*) malloc( N * N * sizeof(float) );

	// Fill the grid with some initial values, and display to stdout. fillGrid() is defined in the helper file.
    fillGrid( hostGrid, N );
    printf( "Original grid (only top-left shown if too large):\n" );
    displayGrid( hostGrid, N );

	//
	// Allocate memory for the grid(s) on the GPU and apply the heat equation as per the instructions.
	//

    // Allocate device buffers for input and output grids.
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &status);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &status);

    // Write the host grid to the device input buffer.
    status = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, N * N * sizeof(float), hostGrid, 0, NULL, NULL);

    // Build the OpenCL program and create the kernel from file "cwk3.cl". The kernel function is named "heat".
    cl_kernel kernel = compileKernelFromFile("cwk3.cl", "heat", context, device);

    // Set the kernel arguments: 0 -> input buffer, 1 -> output buffer, 2 -> grid dimension N.
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &N);

    // Define the NDRange dimensions.
    size_t globalWorkSize[2] = { (size_t)N, (size_t)N };
    // Determine a local work size that divides N. We try factors from min(N,4) downwards.
    int local_dim = 1;
    for (int factor = (N < 4 ? N : 4); factor > 0; factor--) {
        if (N % factor == 0) {
            local_dim = factor;
            break;
        }
    }
    size_t localWorkSize[2] = { (size_t)local_dim, (size_t)local_dim };

    // Enqueue the kernel for execution with 2D NDRange.
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clFinish(queue);

    // Read back the computed grid from the device output buffer into hostGrid.
    status = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, N * N * sizeof(float), hostGrid, 0, NULL, NULL);

    // Release the OpenCL resources used for the kernel execution.
    clReleaseKernel(kernel);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);

    //
    // Display the final result. This assumes that the iterated grid was copied back to the hostGrid array.
    //
    printf( "Final grid (only top-left shown if too large):\n" );
    displayGrid( hostGrid, N );

    //
    // Release all resources.
    //
    clReleaseCommandQueue( queue   );
    clReleaseContext     ( context );

    free( hostGrid );

    return EXIT_SUCCESS;
}
