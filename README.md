# used
BFS 
#include <iostream>
#include <queue>
#include <omp.h>
using namespace std;
const int MAX = 1000;
int graph[MAX][MAX], visited[MAX];
void bfs(int start, int n) {
queue<int> q;
visited[start] = 1;
q.push(start);
while(!q.empty()) {
int curr = q.front();
q.pop();
#pragma omp parallel for shared(graph, visited, q) schedule(dynamic)
for(int i=0; i<n; i++) {
if(graph[curr][i] && !visited[i]) {
visited[i] = 1;
q.push(i);
}
}
}
}
int main() {
int n, start;
cout << "Enter number of vertices: ";
cin >> n;
cout << "Enter adjacency matrix:\n";
for(int i=0; i<n; i++) {
for(int j=0; j<n; j++) {
cin >> graph[i][j];
}
}
cout << "Enter starting vertex: ";
cin >> start;
#pragma omp parallel num_threads(4)
{
bfs(start, n);
}
cout << "BFS traversal: ";
for(int i=0; i<n; i++) {
if(visited[i])
cout << i << " ";
}
cout << endl;
return 0;
} 


DFS 
#include <iostream>
#include <stack>
#include <omp.h>
using namespace std;
const int MAX = 1000;
int graph[MAX][MAX], visited[MAX];
void dfs(int start, int n) {
    stack<int> s;
    s.push(start);
    while(!s.empty()) {
        int curr = s.top();
        s.pop();
        if(!visited[curr]) {
            visited[curr] = 1;
            #pragma omp parallel for shared(graph, visited, s) schedule(dynamic)
            for(int i=0; i<n; i++) {
                if(graph[curr][i] && !visited[i]) {
                    s.push(i);
                }
            }
        }
    }
}
int main() {
    int n, start;
    cout << "Enter number of vertices: ";
    cin >> n;
    cout << "Enter adjacency matrix:\n";
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            cin >> graph[i][j];
        }
    }
    cout << "Enter starting vertex: ";
    cin >> start;
    #pragma omp parallel num_threads(4)
    {
        dfs(start, n);
    }
    cout << "DFS traversal: ";
    for(int i=0; i<n; i++) {
        if(visited[i])
            cout << i << " ";
    }
    cout << endl;
    return 0;
}

BUBBLE  SORT
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void parallelBubbleSort(vector<int> &arr) {
    int n = arr.size();
    int i, j;
    #pragma omp parallel for private(i, j) shared(arr)
    for (i = 0; i < n-1; ++i) {
        for (j = 0; j < n-i-1; ++j) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

int main() {
    vector<int> arr = {64, 25, 12, 22, 11};
    
    cout << "Array before Parallel Bubble Sort: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
    
    parallelBubbleSort(arr);
    
    cout << "Array after Parallel Bubble Sort: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
    
    return 0;
}
MERGE SORT
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void merge(vector<int> &arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; ++i) {
        L[i] = arr[l + i];
    }
    for (int j = 0; j < n2; ++j) {
        R[j] = arr[m + 1 + j];
    }

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i++];
        } else {
            arr[k] = R[j++];
        }
        ++k;
    }

    while (i < n1) {
        arr[k++] = L[i++];
    }
    while (j < n2) {
        arr[k++] = R[j++];
    }
}

void parallelMergeSort(vector<int> &arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);
            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

int main() {
    vector<int> arr = {12, 11, 13, 5, 6, 7};
    
    cout << "Array before Parallel Merge Sort: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
    
    parallelMergeSort(arr, 0, arr.size() - 1);
    
    cout << "Array after Parallel Merge Sort: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
    
    return 0;
}

MIN MAX 
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Parallel Reduction Min Operation
int parallelMin(const vector<int>& arr) {
    int globalMin = arr[0];
    #pragma omp parallel for reduction(min:globalMin)
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < globalMin) {
            globalMin = arr[i];
        }
    }
    return globalMin;
}

// Parallel Reduction Max Operation
int parallelMax(const vector<int>& arr) {
    int globalMax = arr[0];
    #pragma omp parallel for reduction(max:globalMax)
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] > globalMax) {
            globalMax = arr[i];
        }
    }
    return globalMax;
}

// Parallel Reduction Sum Operation
int parallelSum(const vector<int>& arr) {
    int globalSum = 0;
    #pragma omp parallel for reduction(+:globalSum)
    for (size_t i = 0; i < arr.size(); ++i) {
        globalSum += arr[i];
    }
    return globalSum;
}

// Parallel Reduction Average Operation
double parallelAverage(const vector<int>& arr) {
    int sum = parallelSum(arr);
    return static_cast<double>(sum) / arr.size();
}

int main() {
    vector<int> arr = {5, 3, 9, 1, 7, 2, 8, 4, 6};

    // Min Operation
    cout << "Minimum: " << parallelMin(arr) << endl;

    // Max Operation
    cout << "Maximum: " << parallelMax(arr) << endl;

    // Sum Operation
    cout << "Sum: " << parallelSum(arr) << endl;

    // Average Operation
    cout << "Average: " << parallelAverage(arr) << endl;

    return 0;
}

CUDA ADD
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1000000; // Size of vectors
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on host
    std::vector<int> h_a(N);
    std::vector<int> h_b(N);
    std::vector<int> h_c(N);

    // Initialize vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Perform vector addition on device
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back from device to host
    cudaMemcpy(h_c.data(), d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cout << "Error at index " << i << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

MULTIPLY
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMul(int *a, int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024; // Matrix size
    const int threadsPerBlock = 16;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on host
    std::vector<int> h_a(N * N);
    std::vector<int> h_b(N * N);
    std::vector<int> h_c(N * N);

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Perform matrix multiplication on device
    dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
    dim3 blocksPerGridDim(blocksPerGrid, blocksPerGrid);
    matrixMul<<<blocksPerGridDim, threadsPerBlockDim>>>(d_a, d_b, d_c, N);

    // Copy result back from device to host
    cudaMemcpy(h_c.data(), d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the result (optional)

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
