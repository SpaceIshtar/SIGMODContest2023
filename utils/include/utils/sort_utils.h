#include <algorithm>

namespace utils {
    template<typename K, typename V>
    void sort_2_array(K *key, V *val, std::size_t dim) {
        std::sort(val, val+dim, [&key](int a, int b){return key[a] < key[b];});
    }

} // utils
