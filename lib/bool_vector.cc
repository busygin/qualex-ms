/***********************************************************************
!! bool_vector is a simple manager of bit strings allowing a set of   !!
!! basic bitwise operations and optimized iterating through           !!
!! true entries.                                                      !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2000-2017. All rights reserved.   !!
!!                                                                    !!
!! bool_vector implementation                                         !!
***********************************************************************/

#include "bool_vector.h"

// insert a new i-th entry, false, moving the entries from i on up by one
void bool_vector::insert(int i) {
  if(size_%word_bits==0) words_.push_back(0);
  ++size_;
  int k = word_of(i);
  // each word above the k-th takes the top bit of the word below it
  for(int t=n_words()-1;t>k;--t)
    words_[t] = (words_[t] << 1) | (words_[t-1] >> (word_bits-1));
  word below = words_[k] & ((word(1) << bit_of(i))-1);
  words_[k] = ((words_[k] & ~below) << 1) | below;
}

// erase the i-th entry, moving the entries after it down by one
void bool_vector::erase(int i) {
  int k = word_of(i);
  word below_mask = (word(1) << bit_of(i))-1;
  word below = words_[k] & below_mask;
  words_[k] = ((words_[k] >> 1) & ~below_mask) | below;
  // each word above the k-th gives its bottom bit to the word below it
  for(int t=k+1;t<n_words();++t) {
    words_[t-1] |= words_[t] << (word_bits-1);
    words_[t] >>= 1;
  }
  --size_;
  if(size_%word_bits==0) words_.pop_back();
}
