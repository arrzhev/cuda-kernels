#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

template <typename T>
bool compareVectors(const std::vector<T>& v1, const std::vector<T>& v2, float atol = std::numeric_limits<float>::epsilon()) {
    if (v1.size() != v2.size())
        return false;

    return std::equal(v1.begin(), v1.end(), v2.begin(),
         [atol](T a, T b) {
         return std::fabs(a - b) <= atol;
         });
}