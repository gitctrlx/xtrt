export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=72
cmake --build build