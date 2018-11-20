#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <time.h>

#include <stdio.h>
#include <stdbool.h>

#include <CL/cl.h>

#define MAX_SOURCE_SIZE 0x100000

#define THREADS_COUNT 500
#define MATRIX_LEN 2000


void read_matrix(const char* path, int* matrix)
{
	FILE* input_file = fopen(path, "r");
	if (input_file == NULL)
	{
		printf("Error occurred while reading matrix from file\n");
		exit(0);
	}
	int i = 0;
	int num;
	while (fscanf(input_file, "%d ", &num) > 0)
	{
		matrix[i++] = num;
	}
	fclose(input_file);
}

void rand_matrix(int* matrix, int len, bool set_one)
{
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++) {
			int val = rand() % 10;
			if (set_one)
			{
				if (i == j)
				{
					val = 1;
				}
				else
				{
					val = 0;
				}
			}
			matrix[i * len + j] = val;
		}
	}
}

void reset_matrix(int* matrix, int len)
{
	for (int i = 0; i < len * len; i++) {
		matrix[i] = 0;
	}
}

void print_matrix(int* matrix, int len)
{
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++) {
			printf("%d ", matrix[i * len + j]);
		}
		printf("\n");
	}
	printf("\n");
}

bool error(char* err, cl_int code)
{
	if (code != CL_SUCCESS)
	{
		printf(err, (int) code);
		return true;
	}
	return false;
}

char* read_kernel(size_t* source_size)
{
	char fileName[] = "./kernel.cl";
	FILE *fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	char* source_str = (char*) malloc (MAX_SOURCE_SIZE);
	*source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	return source_str;
}

void mult_matrices_on_cpu(const int* left, const int* right, int* res, const int len)
{
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{
			int sum = 0;
			for (int k = 0; k < len; k++)
			{
				sum += left[i * len + k] * right[k * len + j];
			}
			res[i * len + j] = sum;
		}
	}
}

void mult_matrices_on_gpu(const int* left, const int* right, int* res, const int len, const size_t threads_count)
{
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if (error("Failed to get platform ID.", ret)) { goto cleanMemory; }

	cl_device_id device_id = NULL;
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (error("Failed to get device ID.", ret)) { goto cleanMemory; }

	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	if (error("Failed to create command queue %d\n", ret)) { goto cleanMemory; }

	size_t data_size = len * len * sizeof(int);
	cl_mem left_buff, right_buff, res_buff;
	left_buff = right_buff = res_buff = NULL;
	left_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
	right_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
	res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &ret);

	ret = clEnqueueWriteBuffer(command_queue, left_buff, CL_TRUE, 0, data_size, (void*)left, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer(command_queue, right_buff, CL_TRUE, 0, data_size, (void*)right, 0, NULL, NULL);
	if (error("Failed to copy data from host to device: %d\n", ret)) { goto cleanMemory; }

	size_t source_size;
	char* source_str = read_kernel(&source_size);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
	if (error("Failed to create OpenCL program from source %d\n", ret)) { goto cleanMemory; }

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to build program %d\n", (int) ret);
		char build_log[16348];
		clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, sizeof (build_log), build_log, NULL);
		printf ("Error in kernel: %s\n", build_log);
		goto cleanMemory;
	}

	cl_kernel kernel = clCreateKernel(program, "multMatrices", &ret);
	if (error("Failed to create kernel %d\n", ret)) { goto cleanMemory; }

	int step = (int) (len / threads_count);

	ret  = clSetKernelArg(kernel, 0, sizeof (cl_mem), (void*) &left_buff);
	ret |= clSetKernelArg(kernel, 1, sizeof (cl_mem), (void*) &right_buff);
	ret |= clSetKernelArg(kernel, 2, sizeof (cl_mem), (void*) &res_buff);
	ret |= clSetKernelArg(kernel, 3, sizeof (cl_int), (void*) &len);
	ret |= clSetKernelArg(kernel, 4, sizeof (cl_int), (void*) &step);
	if (error("Failed to set kernel arguments %d\n", ret)) { goto cleanMemory; }

	size_t global_work_size, local_work_size;
	local_work_size = 1;
	global_work_size = threads_count;

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	if (error("Failed to execute kernel %d\n", ret)) { goto cleanMemory; }

	ret = clEnqueueReadBuffer(command_queue, res_buff, CL_TRUE, 0, data_size, (void *)res, 0, NULL, NULL);
	if (error("Failed to copy data from device to host %d\n", ret)) { goto cleanMemory; }

cleanMemory:

	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	clReleaseMemObject(left_buff);
	clReleaseMemObject(right_buff);
	clReleaseMemObject(res_buff);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	free(source_str);
}

double elapsed(clock_t start)
{
	return (((double)clock() - start) / CLOCKS_PER_SEC) * 1000;
}

int main(void)
{
	srand((unsigned)time(NULL));

	int len = MATRIX_LEN;

	int* left = (int*) malloc(len * len * sizeof(int));
	int* right = (int*) malloc(len * len * sizeof(int));

	char path[50];
	sprintf(path, "./inputs/matrix_%dx%d", len, len);

	read_matrix(path, left);
	read_matrix(path, right);

	int* result = malloc(len * len * sizeof(int));
	reset_matrix(result, len);

	double cpu_elapsed, gpu_elapsed;

	printf("Matrix size: %dx%d\n", len, len);
	printf("Threads: %d\n\n", THREADS_COUNT);

	printf("Running on CPU...");
	clock_t start = clock();
	mult_matrices_on_cpu(left, right, result, len);
	cpu_elapsed = elapsed(start);
	printf("\nCPU elapsed: %f milliseconds\n\n", cpu_elapsed);
	reset_matrix(result, len);

	printf("Running on GPU...");
	size_t threads_count = THREADS_COUNT;
	start = clock();
	mult_matrices_on_gpu(left, right, result, len, threads_count);
	gpu_elapsed = elapsed(start);
	printf("\nGPU elapsed: %f milliseconds\n\n", gpu_elapsed);

	double acceleration = cpu_elapsed / gpu_elapsed * 1.0;
	printf("Acceleration: %f\n", acceleration);
	printf("Efficiency: %f\n", acceleration / threads_count * 1.0);

//	print_matrix(left, len);
//	print_matrix(right, len);
//	print_matrix(result, len);

	free(left);
	free(right);
	free(result);

	return 0;
}
