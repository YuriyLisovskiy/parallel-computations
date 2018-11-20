#include <mpi.h>

#include <time.h>
#include <string>
#include <fstream>
#include <iostream>

#define ROOT_ID 0

int* read_matrix(const std::string& path, int& matrix_len)
{
	std::ifstream input_file(path.c_str());
	if (input_file.good())
	{
		int i = 0;
		input_file >> matrix_len;
		int size = matrix_len * matrix_len;
		int* numbers = new int[size];
		int current_number;
		while (input_file >> current_number)
		{
			if (i >= size)
			{
				break;
			}
			numbers[i++] = current_number;
		}
		input_file.close();
		return numbers;
	}
	else
	{
		throw std::invalid_argument("Error occurred while reading matrix from file");
	}
}

void write_results(const double sequential_time, const double parallel_time, const int threads_count)
{
	std::ofstream output_file("./computation_results.txt", std::ios::app);
	if (output_file.good())
	{
		output_file << "\nThreads: " << threads_count << "\n\n";
		std::cout << "\nThreads: " << threads_count << "\n\n";
		output_file << "Sequential time: " << sequential_time << " milliseconds\n";
		std::cout << "Sequential time: " << sequential_time << " milliseconds\n";
		output_file << "Parallel time: " << parallel_time << " milliseconds\n";
		std::cout << "Parallel time: " << parallel_time << " milliseconds\n";
		double acceleration = sequential_time / parallel_time * 1.0;
		output_file << "Acceleration: " << acceleration << "\n";
		std::cout << "Acceleration: " << acceleration << "\n";
		double efficiency = acceleration / threads_count * 1.0;
		output_file << "Efficiency: " << efficiency << "\n";
		std::cout << "Efficiency: " << efficiency << "\n\n";
		output_file.close();
	}
	else
	{
		throw std::invalid_argument("Error occurred while writing results to file");
	}
}

void check_status(int status, int thread_id)
{
	if (status != MPI_SUCCESS)
	{
		fprintf(stderr, "Error occurred while running thread with id = %d\n", thread_id);
		MPI_Finalize();
	}
}

void print_matrix(const int* matrix, const int len)
{
	std::cout << '\n';
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{
			std::cout << matrix[i * len + j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

double elapsed(clock_t start)
{
	double res = (((double)clock() - start) / CLOCKS_PER_SEC) * 1000;
	return res;
}

void single_thread(const int* left_matrix, const int* right_matrix, const int matrix_len, int step, const int thread_id, const int rem, const int threads_count)
{
	int status;
	clock_t clock_start = clock();

	int start = step * (thread_id - 1);
	int end = step * thread_id;
	int size = step * matrix_len;

	if (end == matrix_len - rem)
	{
		end += rem;
		size = (step + rem) * matrix_len;
	}
	int* res = new int[size];
	int idx_i = 0;
	for (int i = start; i < end; i++)
	{
		for (int j = 0; j < matrix_len; j++)
		{
			res[idx_i * matrix_len + j] = 0;
			for (int k = 0; k < matrix_len; k++)
			{
				res[idx_i * matrix_len + j] += left_matrix[i * matrix_len + k] * right_matrix[k * matrix_len + j];
			}
		}
		idx_i++;
	}
	double clock_end = elapsed(clock_start);
	status = MPI_Send(&clock_end, 1, MPI_DOUBLE, ROOT_ID, 1, MPI_COMM_WORLD);
	check_status(status, thread_id);

	status = MPI_Send(res, size, MPI_INT, ROOT_ID, 0, MPI_COMM_WORLD);
	check_status(status, thread_id);
}

double main_thread(int* res, const int matrix_len, int step, const int threads_count, const int rem)
{
	double max_thread_time = 0;
	int status;
	for (int i = 1; i < threads_count; i++)
	{
		double curr_thread_time;
		status = MPI_Recv(&curr_thread_time, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		check_status(status, 0);
		if (curr_thread_time > max_thread_time)
		{
			max_thread_time = curr_thread_time;
		}

		int indexI = (i - 1) * step;
		int size = step * matrix_len;
		if (i == threads_count - 1)
		{
			step += rem;
			size = step * matrix_len;
		}
		int* block = new int[size];
		status = MPI_Recv(block, size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		check_status(status, 0);
		for (int j = 0; j < step; j++) {
			for (int k = 0; k < matrix_len; k++)
			{
				res[indexI * matrix_len + k] = block[j * matrix_len + k];
			}
			indexI++;
		}
		delete[] block;
	}
	return max_thread_time;
}

void sequential(const int* left_matrix, const int* right_matrix, const int matrix_len)
{
	int* res = new int[matrix_len * matrix_len];
	for (int i = 0; i < matrix_len; i++)
	{
		for (int j = 0; j < matrix_len; j++)
		{
			res[i * matrix_len + j] = 0;
			for (int k = 0; k < matrix_len; k++)
			{
				res[i * matrix_len + j] += left_matrix[i * matrix_len + k] * right_matrix[k * matrix_len + j];
			}
		}
	}
//	print_matrix(res, matrix_len);
	delete []res;
}

int main(int argc, char **argv) {
	int threads_count;
	int current_thread_id;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &threads_count);
	MPI_Comm_rank(MPI_COMM_WORLD, &current_thread_id);

//	std::string left_path = "./inputs/matrix_10x10";
//	std::string right_path = "./inputs/matrix_10x10";
	std::string left_path = "./inputs/matrix_1000x1000";
	std::string right_path = "./inputs/matrix_1000x1000";

	int matrix_len;
	int* left_matrix = read_matrix(left_path, matrix_len);
	int* right_matrix = read_matrix(right_path, matrix_len);
	int rem = matrix_len % (threads_count - 1);
	int step = (matrix_len - rem) / (threads_count - 1);

	if (current_thread_id != ROOT_ID)
	{
		single_thread(left_matrix, right_matrix, matrix_len, step, current_thread_id, rem, threads_count);
	}
	else
	{
		int* result = new int[matrix_len * matrix_len];
		double parallel_time = main_thread(result, matrix_len, step, threads_count, rem);
	//	print_matrix(result, matrix_len);

		clock_t clock_start = clock();
		sequential(left_matrix, right_matrix, matrix_len);
		double sequential_end = elapsed(clock_start);

		write_results(sequential_end, parallel_time, threads_count);
	}

	MPI_Finalize();
	return 0;
}
