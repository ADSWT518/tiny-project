if [ ! -d "./build" ]; then
    mkdir "build"
fi

cd build
LLVM_DIR=/Path/to/llvm-project/build/lib/cmake/llvm \
MLIR_DIR=/Path/to/llvm-project/lib/cmake/mlir \
cmake -G Ninja ..
cmake --build . --target tiny
cd ..
