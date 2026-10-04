#ifndef VEC_DEL_H
#define VEC_DEL_H

#include <vector>
#include <algorithm>

// vec_del() erases value from the sorted vector x, which must hold it
template<class T>
inline void vec_del(std::vector<T>& x, T value){
  x.erase(std::lower_bound(x.begin(),x.end(),value));
}

#endif  // VEC_DEL_H
