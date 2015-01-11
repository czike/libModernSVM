package com.kno10.svm.libmodernsvm.kernelfunction.unsafe;

/**
 * Radial basis function (RBF) kernel.
 */
public class RadialBasisKernelFunction extends AbstractKernelFunction {
  double gamma;

  public RadialBasisKernelFunction(double gamma) {
    super();
    this.gamma = gamma;
  }

  public double similarity(UnsafeSparseVector x, UnsafeSparseVector y) {
    double sum = UnsafeSparseVector.squareEuclidean(x, y);
    return Math.exp(-gamma * sum);
  }

  public double gamma() {
    return gamma;
  }
}
