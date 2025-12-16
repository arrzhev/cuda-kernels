# Vector addition kernel test sample
add_executable(vectorAdd vectorAdd.cpp)
target_link_libraries(vectorAdd PRIVATE cuda_c)

# Grayscale kernel test sample
add_executable(grayscale grayscale.cpp)
target_link_libraries(grayscale PRIVATE cuda_c)

# Mean blur kernel test sample
add_executable(meanBlur meanBlur.cpp)
target_link_libraries(meanBlur PRIVATE cuda_c)

# Vector dot product kernel test sample
add_executable(vectorDotProduct vectorDotProduct.cpp)
target_link_libraries(vectorDotProduct PRIVATE cuda_c)

# Matrix X Vector multiplication kernel test sample
add_executable(matrixVectorMul matrixVectorMul.cpp)
target_link_libraries(matrixVectorMul PRIVATE cuda_c)

# Matrix X Matrix multiplication kernel test sample
add_executable(matmul matmul.cpp)
target_link_libraries(matmul PRIVATE cuda_c)