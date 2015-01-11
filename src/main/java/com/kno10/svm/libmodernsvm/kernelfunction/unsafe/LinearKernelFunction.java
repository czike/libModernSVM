package com.kno10.svm.libmodernsvm.kernelfunction.unsafe;

/**
 * Linear kernel.
 */
public class LinearKernelFunction extends AbstractKernelFunction {
  public double similarity(UnsafeSparseVector x, UnsafeSparseVector y) {
    return UnsafeSparseVector.dot(x, y);
  }
}
