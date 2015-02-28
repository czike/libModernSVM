package com.kno10.svm.libmodernsvm.kernelfunction.sparsevec;

import com.kno10.svm.libmodernsvm.kernelfunction.Vector;

/**
 * More compact sparse vector type.
 */
public class SparseVector implements Vector<SparseVector> {
  public int[] index;

  public double[] value;

  public SparseVector(int[] index, double[] value) {
    super();
    this.index = index;
    this.value = value;
  }

  @Override
  public int size() {
    return index.length;
  }

  @Override
  public int index(int i) {
    return index[i];
  }

  @Override
  public double value(int i) {
    return value[i];
  }

  @Override
  public double dot(SparseVector y) {
    double sum = 0.;
    final int xlen = this.index.length, ylen = y.index.length;
    int i = 0, j = 0;
    while(i < xlen && j < ylen) {
      int xi = this.index[i], yi = y.index[j];
      if(xi == yi) {
        sum += this.value[i++] * y.value[j++];
      }
      else {
        if(xi > yi) {
          ++j;
        }
        else {
          ++i;
        }
      }
    }
    return sum;
  }

  @Override
  public double squareEuclidean(SparseVector y) {
    double sum = 0.;
    final int xlen = this.index.length, ylen = y.index.length;
    int i = 0, j = 0;
    while(i < xlen && j < ylen) {
      final int xi = this.index[i], yi = y.index[j];
      double d = 0.;
      if(xi <= yi) {
        d += this.value(i++);
      }
      if(yi <= xi) {
        d -= y.value(j++);
      }
      sum += d * d;
    }
    while(i < xlen) {
      final double d2 = this.value[i++];
      sum += d2 * d2;
    }
    while(j < ylen) {
      final double d2 = y.value[j++];
      sum += d2 * d2;
    }
    return sum;
  }
}
