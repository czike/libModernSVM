package com.kno10.svm.libmodernsvm.kernelfunction;


/**
 * Linear kernel.
 */
public class LinearKernelFunction<V extends Vector<V>> extends AbstractKernelFunction<V> {
  @Override
  public double similarity(V x, V y) {
    return x.dot(y);
  }
}
