/******************************************************************************
 *                               IME-USP (2018)                               *
 *             MAC0219 - Programacao Concorrente e Paralela - EP1             *
 *                                                                            *
 *                                   Matriz                                   *
 *                                                                            *
 *                      Marcelo Schmitt   - NUSP 9297641                      *
 *                      Raphael R. Gusmao - NUSP 9778561                      *
 ******************************************************************************/

#ifndef MATRIX_H
#define MATRIX_H

using namespace std;

// Fonte
#define UNDERLINE "\033[4m"    // Underline_
#define CYAN      "\033[36;1m" // Azul claro
#define GREEN     "\033[32;1m" // Verde
#define PINK      "\033[35;1m" // Rosa
#define YELLOW    "\033[33;1m" // Amarelo
#define END       "\033[0m"    // Para de pintar

// Matriz
class Matrix {
public:
    long long **matrix;  // Matriz
    int rows;            // Numero de linhas
    int cols;            // Numero de colunas
    char implementation; // Implementacao utilizada ("c":CPU, "g":GPU)
    /**************************************************************************/
    Matrix (int _rows, int _cols, char _implementation); // Construtor
    void show ();                                        // Exibe a matriz
};

// Retorna o tempo atual em nanossegundos
uint64_t get_time ();

// Le o arquivo que contem a lista de matrizes
vector<Matrix> read_matrix_list (char *path, char implementation);

// Calcula o minimo entre dois numeros a e b (evitando Branch Divergence)
#define minimum(a, b) (b ^ ((a ^ b) & -(a < b)))

// Devolve a matriz M = min(matrix_list)
Matrix MATRIX_reduction (vector<Matrix> matrix_list, char implementation, bool debug);

// Calcula a matriz M = min(matrix_list) sequencialmente
void sequential_reduction (vector<Matrix> matrix_list, Matrix *M);

// Calcula a matriz M = min(matrix_list) utilizando o CUDA
__global__
void cuda_reduction (long long *matrix_list, int size, int rows, int cols);

#endif

/******************************************************************************/
