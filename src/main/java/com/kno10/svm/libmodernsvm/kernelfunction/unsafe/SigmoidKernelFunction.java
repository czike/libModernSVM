package com.kno10.svm.libmodernsvm.kernelfunction.unsafe;

/**
 * Sigmoid kernel.
 */
public class SigmoidKernelFunction extends AbstractKernelFunction {
  private final double gamma;

  private final double coef0;

  public SigmoidKernelFunction(double gamma, double coef0) {
    super();
    this.gamma = gamma;
    this.coef0 = coef0;
  }

  public double similarity(UnsafeSparseVector x, UnsafeSparseVector y) {
    return Math.tanh(gamma * UnsafeSparseVector.dot(x, y) + coef0);
  }
}
