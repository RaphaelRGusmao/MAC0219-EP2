################################################################################
#                                IME-USP (2018)                                #
#              MAC0219 - Programacao Concorrente e Paralela - EP2              #
#                                                                              #
#                                    Testes                                    #
#                                                                              #
#                       Marcelo Schmitt   - NUSP 9297641                       #
#                       Raphael R. Gusmao - NUSP 9778561                       #
################################################################################

import random
import sys

class Color:
    CYAN   = '\033[36;1m'
    GREEN  = '\033[32;1m'
    YELLOW = '\033[33;1m'
    END    = '\033[0m'

################################################################################
def matrix_read (path):
    file = open(path, "r")
    matrix = [map(int, i) for i in [x.split() for x in file.readlines()]]
    file.close()
    return matrix

################################################################################
def matrix_list_read (path):
    file = open(path, "r")
    lines = [x.split() for x in file.readlines()]
    n = int(lines[0][0])
    matrix_list = []
    for i in range(n):
        matrix_list.append([map(int, i) for i in [lines[i+2 + i*3], lines[i+3 + i*3], lines[i+4 + i*3]]])
    file.close()
    return matrix_list

################################################################################
def matrix_list_save (matrix_list, path):
    file = open(path, "w")
    file.write(str(len(matrix_list)) + "\n")
    for M in matrix_list:
        file.write("***\n")
        for i in range(len(M)):
            for j in range(len(M[0])):
                    file.write(str(M[i][j]) + " ")
            file.write("\n")
    file.write("***\n")
    file.close()

################################################################################
def matrix_list_random (n, rows, cols, min, max):
    matrix_list = [[[0 for j in range(cols)] for i in range(rows)] for k in range(n)]
    for k in range(n):
        for i in range(rows):
            for j in range(cols):
                matrix_list[k][i][j] = random.randint(min, max)
    return matrix_list

################################################################################
def check ():
    print(Color.CYAN + "\nLendo a lista de matrizes..." + Color.END)
    matrix_list = matrix_list_read("test_py.txt")
    print(Color.CYAN + "Lendo resposta..." + Color.END)
    try:
        ans = matrix_read("ans")
    except:
        print(Color.YELLOW + "Salve a resposta do EP2 em um arquivo \"ans\" antes!" + Color.END)
        return
    print(Color.CYAN + "Calculando resultado correto..." + Color.END)
    r_matrix = matrix_list[0]
    for k in range (1, len(matrix_list)):
        for i in range (len(matrix_list[0])):
            for j in range (len(matrix_list[0][0])):
                r_matrix[i][j] = min(r_matrix[i][j], matrix_list[k][i][j])
    if (ans == r_matrix):
        print(Color.GREEN + "Tudo certo!\n" + Color.END)
    else:
        print(Color.YELLOW + "Algum valor errado!\n" + Color.END)

################################################################################
def main ():
    if (len(sys.argv) == 2):
        matrix_list_save(matrix_list_random(int(sys.argv[1]), 3, 3, 0, 10000), "test_py.txt")
        print(Color.CYAN + "Lista de matrizes gerada!\n" + Color.END)
    else:
        check()

main()

################################################################################
