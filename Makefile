CC=nvcc
CFLAGS=-lm -O2 -w -D Tile_Width=128 -arch sm_30
DEVICE_ID=0

all: ep2

ep2.o: ep2.cu
	$(CC) $(CFLAGS) -c ep2.cu

ep2: ep2.o
	$(CC) $(CFLAGS) ep2.o -o ep2

run:
	./ep2 10 1

clean:
	rm -rf *.o ep2
