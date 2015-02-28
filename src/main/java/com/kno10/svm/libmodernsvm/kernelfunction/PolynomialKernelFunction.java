package com.kno10.svm.libmodernsvm.kernelfunction;


/**
 * Polynomial kernel.
 */
public class PolynomialKernelFunction<V extends Vector<V>> extends AbstractKernelFunction<V> {
  /** Kernel degree */
  private final int degree;

  /** Gamma factor */
  private final double gamma;

  /** Offset coefficient */
  private final double coef0;

  public PolynomialKernelFunction(int degree, double gamma, double coef0) {
    super();
    this.degree = degree;
    this.gamma = gamma;
    this.coef0 = coef0;
  }

  @Override
  public double similarity(V x, V y) {
    return powi(gamma * x.dot(y) + coef0, degree);
  }

  public int degree() {
    return degree;
  }

  public double gamma() {
    return gamma;
  }

  public double coeff0() {
    return coef0;
  }
}
