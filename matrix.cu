/******************************************************************************
 *                               IME-USP (2018)                               *
 *             MAC0219 - Programacao Concorrente e Paralela - EP1             *
 *                                                                            *
 *                                   Matriz                                   *
 *                                                                            *
 *                      Marcelo Schmitt   - NUSP 9297641                      *
 *                      Raphael R. Gusmao - NUSP 9778561                      *
 ******************************************************************************/

#include <bits/stdc++.h>
#include "matrix.h"
using namespace std;

Matrix::Matrix (int _rows, int _cols, char _implementation)
{
    rows = _rows;
    cols = _cols;
    implementation = _implementation;
    matrix = new long long*[rows];
    matrix[0] = (long long*)calloc(rows*cols, sizeof(long long));
    for (int i = 1; i < rows; i++) {
        matrix[i] = matrix[0] + i*cols;
    }
}

/******************************************************************************/
void Matrix::show ()
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
}

/******************************************************************************/
vector<Matrix> read_matrix_list (char *path, char implementation)
{
    ifstream file(path);
    if (!file.is_open()) {
        cout << YELLOW << "Error reading file (" << path << ")" << END << endl;
        exit(1);
    }
    int size; file >> size;
    vector<Matrix> matrix_list;
    for (int i = 0; i < size; i++) {
        string ignore; file >> ignore; // ***
        Matrix M(3, 3, implementation);
        file >> M.matrix[0][0] >> M.matrix[0][1] >> M.matrix[0][2];
        file >> M.matrix[1][0] >> M.matrix[1][1] >> M.matrix[1][2];
        file >> M.matrix[2][0] >> M.matrix[2][1] >> M.matrix[2][2];
        matrix_list.push_back(M);
    }
    file.close();
    return matrix_list;
}

/******************************************************************************/
uint64_t get_time ()
{
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return (uint64_t)(time.tv_sec)*1000000000 + (uint64_t)(time.tv_nsec);
}

/******************************************************************************/
Matrix MATRIX_reduction (vector<Matrix> matrix_list, char implementation, bool debug)
{
    uint64_t start, end;
    Matrix M(matrix_list[0].rows, matrix_list[0].cols, implementation);
    if (implementation == 'c') { // CPU
        if (debug) cout << CYAN << UNDERLINE << "CPU" << END << endl;
        memcpy(M.matrix[0], matrix_list[0].matrix[0], M.rows*M.cols*sizeof(long long));
        // Executa o codigo na CPU
        start = get_time();
            sequential_reduction(matrix_list, &M);
        end = get_time();
    } else { // GPU
        if (debug) cout << CYAN << UNDERLINE << "GPU" << END << endl;
        // Transformando a lista de matriz em um formato compativel com o CUDA
        long long *matrix_list_cuda;
        cudaMalloc(&matrix_list_cuda, matrix_list.size()*M.rows*M.cols*sizeof(long long));
        for (int k = 0; k < matrix_list.size(); k++) {
            cudaMemcpy(matrix_list_cuda + k*M.rows*M.cols,
                       matrix_list[k].matrix[0],M.rows*M.cols*sizeof(long long),
                       cudaMemcpyHostToDevice);
        }

        // Executa o codigo na GPU
        int block_size = 128;
        int n_blocks = (matrix_list.size()/2 + block_size - 1) / block_size;
        start = get_time();
            cuda_reduction<<<n_blocks, block_size>>>(matrix_list_cuda, matrix_list.size(), M.rows, M.cols);
            cudaDeviceSynchronize();
        end = get_time();

        // Salva o resultado na matriz M
        long long *M_host = new long long[M.rows*M.cols];
        cudaMemcpy(M_host, matrix_list_cuda, M.rows*M.cols*sizeof(long long), cudaMemcpyDeviceToHost);
        int index = 0;
        for (int i = 0; i < M.rows; i++) {
            for (int j = 0; j < M.cols; j++) {
                M.matrix[i][j] = M_host[index++];
            }
        }

        // Libera a memoria
        cudaFree(matrix_list_cuda);
        free(M_host);
    }
    if (debug) cout << CYAN << "Reduction time: " << GREEN << (double)(end-start)/1000000000 << " s" << endl;
    return M;
}

/******************************************************************************/
void sequential_reduction (vector<Matrix> matrix_list, Matrix *M)
{
    for (int k = 1; k < matrix_list.size(); k++) {
        for (int i = 0; i < matrix_list[0].rows; i++) {
            for (int j = 0; j < matrix_list[0].cols; j++) {
                M->matrix[i][j] = minimum(M->matrix[i][j], matrix_list[k].matrix[i][j]);
            }
        }
    }
}

/******************************************************************************/
__global__
void cuda_reduction (long long *matrix_list, int size, int rows, int cols)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int n = size;
    while (n > 1) {
        if (thread_id == 0 && n % 2 == 1) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int a_index =                   i*cols + j;
                    int b_index = (n-1)*rows*cols + i*cols + j;
                    matrix_list[a_index] = minimum(matrix_list[a_index],
                                                   matrix_list[b_index]);
                }
            }
        }
        __syncthreads();
        n /= 2;
        if (thread_id < n) {
            for (int k = 0; k < n; k++) {
                if (thread_id == k) {
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            int a_index =     k*rows*cols + i*cols + j;
                            int b_index = (k+n)*rows*cols + i*cols + j;
                            matrix_list[a_index] = minimum(matrix_list[a_index],
                                                           matrix_list[b_index]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

/******************************************************************************/
