package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.ArrayUtil;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
/**
 * This is the original cache from the libSVN implementation. The code is very C
 * stylish, and probably not half as effective on Java due to garbage
 * collection.
 */
class Cache<T> {
  private final int l;

  private long size;

  private static final class head_t {
    head_t prev, next; // a circular list

    float[] data;

    int len; // data[0,len) is cached in this entry
  }

  private final head_t[] head;

  private head_t lru_head;

  private final DataSet<T> x;

  private final KernelFunction<? super T> kf;

  public Cache(DataSet<T> x_, KernelFunction<? super T> kf_, long size_) {
    this.x = x_;
    this.kf = kf_;
    this.l = x.size();
    head = new head_t[l];
    for(int i = 0; i < l; i++) {
      head[i] = new head_t();
    }
    size = size_ >> 2;
    size -= l * (24 >> 2); // In Java, we need 24 bytes for the head object!
    // Minimum cache size is two columns:
    size = Math.max(size, 2 * (long) l);
    lru_head = new head_t();
    lru_head.next = lru_head.prev = lru_head;
  }

  private void lru_delete(head_t h) {
    // delete from current location
    h.prev.next = h.next;
    h.next.prev = h.prev;
  }

  private void lru_insert(head_t h) {
    // insert to last position
    h.next = lru_head;
    h.prev = lru_head.prev;
    h.prev.next = h;
    h.next.prev = h;
  }

  // request data [0,len)
  // return some position p where [p,len) need to be filled
  // (p >= len if nothing needs to be filled)
  // java: simulate pointer using single-element array
  int get_data(int index, float[][] data, int len) {
    head_t h = head[index];
    if(h.len > 0) {
      lru_delete(h);
    }
    int more = len - h.len;

    int ret = h.len;
    if(more > 0) {
      // free old space
      while(size < more) {
        head_t old = lru_head.next;
        lru_delete(old);
        size += old.len;
        old.data = null;
        old.len = 0;
      }

      // allocate new space
      float[] new_data = new float[len];
      if(h.data != null) {
        System.arraycopy(h.data, 0, new_data, 0, h.len);
      }
      h.data = new_data;
      h.len = len;
      size -= more;
    }

    lru_insert(h);
    data[0] = h.data;
    return ret;
  }

  void swap_index(int i, int j) {
    if(i == j) {
      return;
    }
    // Swap in data set:
    x.swap(i, j);
    // Swap in index:
    ArrayUtil.swap(head, i, j);

    // Ensure i < j
    if(i > j) {
      int tmp = i;
      i = j;
      j = tmp;
    }
    // Swap in cached lists:
    for(head_t h = lru_head.next; h != lru_head; h = h.next) {
      if(h.len > i) {
        if(h.len > j) {
          ArrayUtil.swap(h.data, i, j);
        }
        else {
          // Discard this cache:
          lru_delete(h);
          size += h.len;
          h.data = null;
          h.len = 0;
        }
      }
    }
  }

  public double similarity(int i, int j) {
    return kf.similarity(x.get(i), x.get(j));
  }
}