#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

#include <limits.h>
//#define Tile_Width = 64



// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__inline__ __device__
int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}


//Modificacao para a reducao do EP2
__inline__ __device__
void warpReduceMin(int& val, int& idx) {
  int other_val;
  for (int offset = warpSize/2; offset > 0; offset /= 2) { 
    //int other_val = __shfl_down(val, offset);
    //other_val = __shfl_down(val, offset);
    //val = other_val < val ? other_val : val;
    //lowest = __shfl_down(lowest, offset);
    //val += __shfl_down(val, offset);
    int tmpVal = __shfl_down(val, offset);
    int tmpIdx = __shfl_down(idx, offset);
    if (tmpVal < val) {
        val = tmpVal;
        idx = tmpIdx;
    }
  }
  //return val;
}


__inline__ __device__
void blockReduceMin(int& val, int& idx) {

  static __shared__ int shared[32], indices[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  warpReduceMin(val, idx);     // Each warp performs partial reduction

  if (lane==0) {
    shared[wid]=val; // Write reduced value to shared memory
    indices[wid] = idx;
  } 

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared[lane];
    idx = indices[lane];
  } else {
    val = INT_MAX;
    idx = 0;
  }


  if (wid==0) warpReduceMin(val, idx); //Final reduce within first warp

  //return val;
}


// Funcao para fazer a reducao do EP2 usando o exemplo desse post:
//https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
//https://stackoverflow.com/questions/41996828/cuda-reduction-minimum-value-and-index
__global__ void deviceReduceKernel(int *in, int* out, int N) {
  int sum = INT_MAX;
  int minIdx = 0;
  //int sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    //sum = in[i] < sum ? in[i] : sum;
    //sum += in[i];
    if (in[i] < sum) {
      sum = in[i];
      minIdx = i;
    }
  }
  blockReduceMin(sum, minIdx);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

void deviceReduce(int *in, int* out, int N) {
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  deviceReduceKernel<<<blocks, threads>>>(in, out, N);
  deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

//  printf("Execute the kernel...\n");

//  int GridSize = (N + Tile_Width-1) / Tile_Width;
//  dim3 gridDim(GridSize, GridSize);
  //dim3 blockDim(Tile_Width, Tile_Width, 2);
//  dim3 blockDim(3, 3, 2);

 // matSum<<< gridDim, blockDim >>>(dev_S, dev_A, dev_B, N);
//  matMinRed<<< gridDim, blockDim >>>(dev_S, dev_A, dev_B, 3, 3, N);

__global__ void ep2matReduce(int* S, int* A, int* B, int N) {

  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = i*N + j;

  ///int* result = new int[size1 + size2];
  //copy(arr1, arr1 + size1, result);
  //copy(arr2, arr2 + size2, result + size1);


  if (tid < N*N) {
    S[tid] = A[tid] + B[tid];
  }
}


__global__ void matRedux(int* S, int* A, int* B, int N) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = i*N + j;
  if (tid < N*N) {
    S[tid] = A[tid] + B[tid];
  }
}


__global__ void matMinRed(int* S, int* A, int* B, int dim1, int dim2, int N) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = i*dim1 + j;
  if (tid < dim1*dim2) {
    int menorA = INT_MAX;
    int menorB = INT_MAX;
    int vector_index = i*dim1 + j*dim2;
    if (threadIdx.z == 0) {
        //printf("menorA: %d\n", menorA);
        for (int k = vector_index; i < vector_index + N; i++) //paca cada elemento do vetor a_ij
          if (A[k] < menorA) menorA = A[k];
    //} else {//threadIdx.z == 1
        //printf("menorB: %d\n", menorB);
    //  for (int k = vector_index; k < vector_index + N; k++) //para cada elemento do vetor b_ij
    //      if (B[k] < menorB) menorB = B[k];
    }
    //TODO provavelmente vai ter que sincronizar as threads threadIdx.z == 0  e threadIdx.z == 1
    if (threadIdx.z == 0)
        S[vector_index] = (menorA < menorB) ? menorA : menorB;

    //S[tid] = A[tid] + B[tid];
  }
}

__global__ void matSum(float* S, float* A, float* B, int N) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = i*N + j;
  if (tid < N*N) {
    S[tid] = A[tid] + B[tid];
  }
}


// Fills a vector with random float entries.
void randomInit(float* data, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int tid = i*N+j;
      data[tid] = (float)drand48();
    }
  }
}

//Fills a vector with random int entries
void randomIntInit(int* data, int d1, int d2, int d3) {
  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      for (int k = 0; k < d3; k++) {
        int index = i*d1 + j*d2 + k;
        //int numero = rand() % 100;
        data[index] = rand() % 100;
        //printf("index[%d] = %d\n", index, numero);
      }
    }
  }
} 

//Print 3D int vector
void print3DintVector(int* data, int d1, int d2, int d3) {
  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      for (int k = 0; k < d3; k++) {
        int index = i*d1 + j*d2 + k;
        printf("%d, ", data[index]);
      }
      printf("\t");
    }
    printf("\n");
  }
}

int main(int argc, char* argv[])
{

  if (argc != 3) {
    fprintf(stderr, "Syntax: %s <matrix size>  <device> \n", argv[0]);
    return EXIT_FAILURE;
  }

  //int Tile_Width = 64;


  int N = atoi(argv[1]);
  int devId = atoi(argv[2]);

  checkCuda( cudaSetDevice(devId) );
  cudaDeviceReset();

  // set seed for drand48()
  //srand48(42);

  printf("INT_MAX: %d\n", INT_MAX);

///////////////
  int* teste = (int*)malloc(10 * sizeof(int));
  randomIntInit(teste, 1, 1, 10);
  for (int i = 0; i < 10; i++)
    printf("teste[%d]: %d\n", i, teste[i]);

  int* resultado = (int*)malloc(10 * sizeof(int));

  int* dev_teste = NULL;
  int* dev_resultado = NULL;

  checkCuda( cudaMalloc((void**) &dev_teste, 10 * sizeof(int)) );
  checkCuda( cudaMalloc((void**) &dev_resultado, 10 * sizeof(int)) );

  checkCuda( cudaMemcpy(dev_teste, teste, 10*sizeof(int), cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(dev_resultado, resultado, 10*sizeof(int), cudaMemcpyHostToDevice) );

  printf("vai CUDA\n");

  deviceReduce(dev_teste, dev_resultado, 10);

  checkCuda( cudaMemcpy( resultado, dev_resultado, 10 * sizeof(int),cudaMemcpyDeviceToHost) );

  for (int i = 0; i < 10; i++)
    printf("resultado[%d]: %d\n", i, resultado[i]);

  free(teste);
  free(resultado);
  checkCuda( cudaFree(dev_resultado) );

/////////////


  // allocate host memory for matrices A and B
  printf("Allocate host memory for matrices A and B...\n");
  //float* A = (float*) malloc(N * N * sizeof(float));
  //float* B = (float*) malloc(N * N * sizeof(float));
  //float* S = (float*) malloc(N * N * sizeof(float));

  int* A = (int*) malloc(3 * 3 * N * sizeof(int));
  int* B = (int*) malloc(3 * 3 * N * sizeof(int));
  int* S = (int*) malloc(3 * 3 * sizeof(int));

  // initialize host matrices
  printf("Initialize host matrices...\n");
  //randomInit(A, N);
  //randomInit(B, N);
  randomIntInit(A, 3, 3, N);
  randomIntInit(B, 3, 3, N);

  // allocate device matrices (linearized)
  //printf("Allocate device matrices (linearized)...\n");
  //float* dev_A = NULL; 
  //float* dev_B = NULL;
  //float* dev_S = NULL;
  //checkCuda( cudaMalloc((void**) &dev_A, N * N * sizeof(float)) );
  //checkCuda( cudaMalloc((void**) &dev_B, N * N * sizeof(float)) );
  //checkCuda( cudaMalloc((void**) &dev_S, N * N * sizeof(float)) );

  int* dev_A = NULL;
  int* dev_B = NULL;
  int* dev_S = NULL;    
  checkCuda( cudaMalloc((void**) &dev_A, 3 * 3 * N * sizeof(int)) );
  checkCuda( cudaMalloc((void**) &dev_B, 3 * 3 * N * sizeof(int)) );
  checkCuda( cudaMalloc((void**) &dev_S, 3 * 3 * sizeof(int)) );

  printf("MatrixA: \n");
  print3DintVector(A, 3, 3, N);
  printf("MatrixB: \n");
  print3DintVector(B, 3, 3, N);


  struct timeval begin, end;
  gettimeofday(&begin, NULL);
  // copy host memory to device
  checkCuda( cudaMemcpy(dev_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(dev_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice) );

  // execute the kernel
  printf("Execute the kernel...\n");

  int GridSize = (N + Tile_Width-1) / Tile_Width;
  dim3 gridDim(GridSize, GridSize);
  //dim3 blockDim(Tile_Width, Tile_Width, 2);
  dim3 blockDim(3, 3, 2);

 // matSum<<< gridDim, blockDim >>>(dev_S, dev_A, dev_B, N);
  matMinRed<<< gridDim, blockDim >>>(dev_S, dev_A, dev_B, 3, 3, N);
  
  checkCuda( cudaDeviceSynchronize() );

  // copy result from device to host
  checkCuda( cudaMemcpy( S, dev_S, N * N * sizeof(float),cudaMemcpyDeviceToHost) );
  gettimeofday(&end, NULL);
  float gpuTime = 1000000*(float)(end.tv_sec - begin.tv_sec);
  gpuTime +=  (float)(end.tv_usec - begin.tv_usec);
  // print times
  printf("\nExecution Time (microseconds): %9.2f\n\n", gpuTime);

  print3DintVector(S, 3, 3, 1);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId) );
  printf("Device: %s\n", prop.name);

  // clean up memory
  free(A);
  free(B);
  free(S);
  checkCuda( cudaFree(dev_A) );
  checkCuda( cudaFree(dev_B) );
  checkCuda( cudaFree(dev_S) );

  return 0;
}

