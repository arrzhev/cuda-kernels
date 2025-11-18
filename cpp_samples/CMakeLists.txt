# Vector addition kernel test sample
add_executable(vectorAdd vectorAdd.cpp)
target_link_libraries(vectorAdd PRIVATE cuda_ops)

# Grayscale kernel test sample
add_executable(grayscale grayscale.cpp)
target_link_libraries(grayscale PRIVATE cuda_ops)

# Mean blur kernel test sample
add_executable(meanBlur meanBlur.cpp)
target_link_libraries(meanBlur PRIVATE cuda_ops)

# Matrix multiplication kernel test sample
add_executable(matrixMul matrixMul.cpp)
target_link_libraries(matrixMul PRIVATE cuda_ops)