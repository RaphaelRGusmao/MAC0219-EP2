################################################################################
#                                IME-USP (2018)                                #
#              MAC0219 - Programacao Concorrente e Paralela - EP2              #
#                                                                              #
#                                   Makefile                                   #
#                                                                              #
#                       Marcelo Schmitt   - NUSP 9297641                       #
#                       Raphael R. Gusmao - NUSP 9778561                       #
################################################################################

.PHONY: clean
CC = nvcc
CFLAGS = -lm -O2 -w -Xptxas --opt-level=3 -arch sm_30
OBJS = \
	matrix.o \
	main.o

all: main

main: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@
	make clean

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *~

run:
	clear
	make
	clear
	./main "test_py.txt" g d

################################################################################
