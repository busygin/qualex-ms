/***********************************************************************
!! bool_vector is a simple manager of bit strings allowing a set of   !!
!! basic bitwise operations and optimized iterating through           !!
!! true entries.                                                      !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2000-2017. All rights reserved.   !!
!!                                                                    !!
!! bool_vector header                                                 !!
***********************************************************************/

#ifndef BOOL_VECTOR_H
#define BOOL_VECTOR_H

#include <bit>
#include <cstdint>
#include <vector>

// bool_vector holds a bit string of a fixed size in 64-bit words.  The bits of
// the last word beyond the size are always zero, which code working on the
// words directly (through words()) must keep so.
class bool_vector {
public:
  using word = std::uint64_t;
  static constexpr int word_bits = 64;

  class one_iterator;
  class one_range;

  bool_vector() = default;

  // a vector of a given size, all entries false
  explicit bool_vector(int size): size_(size), words_(words_for(size), 0) {}

  int size() const { return size_; }

  // i-th entry
  bool at(int i) const { return (words_[word_of(i)] >> bit_of(i)) & 1; }

  // set i-th entry to true
  void put(int i) { words_[word_of(i)] |= word(1) << bit_of(i); }

  // set i-th entry to false
  void clear(int i) { words_[word_of(i)] &= ~(word(1) << bit_of(i)); }

  // insert a new i-th entry, false, moving the entries from i on up by one
  void insert(int i);

  // erase the i-th entry, moving the entries after it down by one
  void erase(int i);

  // the indices of the true entries in increasing order:
  //   for(int j : v.ones()) ...
  one_range ones() const;

  // the words holding the vector, entry i being bit i%word_bits of word
  // i/word_bits
  int n_words() const { return (int)words_.size(); }
  word* words() { return words_.data(); }
  const word* words() const { return words_.data(); }

  bool operator==(const bool_vector& x) const = default;

  // the word holding entry i and the bit of that word; entries are never
  // negative, and unsigned arithmetic makes these a shift and a mask
  static int word_of(int i) { return (unsigned)i/word_bits; }
  static int bit_of(int i) { return (unsigned)i%word_bits; }

private:
  static int words_for(int size) { return (size+word_bits-1)/word_bits; }

  int size_ = 0;
  std::vector<word> words_;
};

// one_iterator walks the true entries of a bool_vector: a zero word is
// skipped whole, and each true entry of a word is found in one step
class bool_vector::one_iterator {
public:
  one_iterator(const word* w, const word* last): w_(w), last_(last) {
    if(w_==last_) return;
    bits_ = *w_;
    skip_zero_words();
  }

  int operator*() const { return base_+std::countr_zero(bits_); }

  one_iterator& operator++() {
    bits_ &= bits_-1;  // drop the entry just visited
    skip_zero_words();
    return *this;
  }

  bool operator==(const one_iterator& x) const {
    return w_==x.w_ && bits_==x.bits_;
  }

private:
  // move on to the next word with a true entry, or to the end
  void skip_zero_words() {
    while(bits_==0) {
      if(++w_==last_) return;
      bits_ = *w_;
      base_ += word_bits;
    }
  }

  const word* w_;
  const word* last_;
  word bits_ = 0;  // the entries of *w_ not visited yet
  int base_ = 0;   // the index of bit 0 of *w_
};

class bool_vector::one_range {
public:
  one_range(const word* first, const word* last): first_(first), last_(last) {}
  one_iterator begin() const { return one_iterator(first_,last_); }
  one_iterator end() const { return one_iterator(last_,last_); }

private:
  const word* first_;
  const word* last_;
};

inline bool_vector::one_range bool_vector::ones() const {
  return one_range(words_.data(),words_.data()+words_.size());
}

#endif  // BOOL_VECTOR_H
