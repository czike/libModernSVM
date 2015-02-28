package com.kno10.svm.libmodernsvm.kernelfunction;


/**
 * Sigmoid kernel.
 */
public class SigmoidKernelFunction<V extends Vector<V>> extends AbstractKernelFunction<V> {
  private final double gamma;

  private final double coef0;

  public SigmoidKernelFunction(double gamma, double coef0) {
    super();
    this.gamma = gamma;
    this.coef0 = coef0;
  }

  @Override
  public double similarity(V x, V y) {
    return Math.tanh(gamma * x.dot(y) + coef0);
  }
}
