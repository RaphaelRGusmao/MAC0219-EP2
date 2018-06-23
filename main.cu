/******************************************************************************
 *                               IME-USP (2018)                               *
 *             MAC0219 - Programacao Concorrente e Paralela - EP2             *
 *                                                                            *
 *                                 Principal                                  *
 *                                                                            *
 *                      Marcelo Schmitt   - NUSP 9297641                      *
 *                      Raphael R. Gusmao - NUSP 9778561                      *
 ******************************************************************************/

#include <bits/stdc++.h>
#include <stdint.h>
#include "matrix.h"
using namespace std;

/******************************************************************************/
// Funcao principal
// Argumentos:
// matrix_list_path: arquivo contendo todas as matrizes 3x3
// implementation (opcional): "c":CPU ou "g":GPU (padrao)
// debug (opcional): "d":modo debug (desativado por padrao)
int main (int argc, char **argv)
{
    /*------------------------------------------------------------------------*/
    if (argc != 2 && argc != 3 && argc != 4) {
        cout << "Usage: ./main <matrix_list_path> [implementation] [debug]" << endl;
        return 1;
    }
    char *matrix_list_path = argv[1];
    char implementation = 'g'; if (argc > 2) implementation = argv[2][0];
    bool debug = false; if (argc > 3 && argv[3][0] == 'd') debug = true;
    if (implementation == 'g') cudaSetDevice(1);

    /*------------------------------------------------------------------------*/
    if (debug) cout << CYAN << "Reading inputs..." << END << endl;
        vector<Matrix> matrix_list = read_matrix_list(matrix_list_path, implementation);
    if (debug) cout << CYAN << "Size: " << matrix_list.size() << END << endl << endl;

    /*------------------------------------------------------------------------*/
    if (debug) cout << CYAN << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[ START ]" << endl;
        Matrix M = MATRIX_reduction(matrix_list, implementation, debug);
    if (debug) cout << CYAN << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[ END ]" << END << endl << endl;

    /*------------------------------------------------------------------------*/
    if (debug) cout << GREEN;
        M.show();
    if (debug) cout << END;

    return 0;
}

/******************************************************************************/
