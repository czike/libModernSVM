package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.ArrayUtil;

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
public abstract class Cache<T> {
  private static final long MEGABYTES = 1 << 20;

  private long size;

  private static final class head_t {
    head_t prev, next; // a circular list

    float[] data;

    int len; // data[0,len) is cached in this entry
  }

  private final head_t[] head;

  private head_t lru_head;

  public Cache(int l, double cache_size) {
    this(l, (long) (cache_size * MEGABYTES));
  }

  public Cache(int l, long size_) {
    head = new head_t[l];
    for(int i = 0; i < l; i++) {
      head[i] = new head_t();
    }
    // 8 bytes chaining, 4 bytes len, 4 bytes data ref,
    // + 4 bytes in head array + 8 bytes Java object overhead
    size = size_ - l * 28;
    size >>= 2; // Bytes to floats.
    // Minimum cache size is two columns:
    size = Math.max(size, 2 * l);
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

  float[] get_data(int index, int len) {
    head_t h = head[index];
    if(h.len > 0) {
      lru_delete(h);
    }
    final int more = len - h.len;
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
      // Compute missing distances:
      for(int j = h.len; j < len; ++j) {
        new_data[j] = (float) similarity(index, j);
      }
      h.data = new_data;
      h.len = len;
      size -= more;
    }

    if(h.len > 0) {
      lru_insert(h);
    }
    return h.data;
  }

  void swap_index(int i, int j) {
    if(i == j) {
      return;
    }
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

  abstract public double similarity(int i, int j);
}