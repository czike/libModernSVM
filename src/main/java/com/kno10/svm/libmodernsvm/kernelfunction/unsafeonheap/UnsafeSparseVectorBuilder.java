package com.kno10.svm.libmodernsvm.kernelfunction.unsafeonheap;

import java.util.Arrays;

import com.kno10.svm.libmodernsvm.kernelfunction.VectorBuilder;

/**
 * Build off-heap vectors.
 * 
 * TODO: sort if values did not arrive in proper order.
 * 
 * @author Erich Schubert
 */
public class UnsafeSparseVectorBuilder implements VectorBuilder<UnsafeSparseVector> {
  /**
   * Initial size of buffers.
   */
  private static final int INITIAL_SIZE = 100;

  /**
   * Indexes of nonzero positions.
   */
  int[] idx = new int[INITIAL_SIZE];

  /**
   * Vector values.
   */
  double[] val = new double[INITIAL_SIZE];

  /**
   * Number of nonzero values.
   */
  int size = 0;

  @Override
  public void clear() {
    size = 0;
  }

  @Override
  public void add(int idx, double val) {
    assert (size == 0 || this.idx[size - 1] < idx) : "Currently, objects must be built SORTED.";
    if(val == 0.) {
      return; // Noop.
    }
    if(size == this.idx.length) { // Resize buffer.
      this.idx = Arrays.copyOf(this.idx, size << 1);
      this.val = Arrays.copyOf(this.val, size << 1);
    }
    this.idx[size] = idx;
    this.val[size] = val;
    ++size;
  }

  @Override
  public UnsafeSparseVector build() {
    return new UnsafeSparseVector(idx, val, size);
  }
}
