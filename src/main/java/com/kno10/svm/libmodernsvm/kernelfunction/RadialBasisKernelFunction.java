package com.kno10.svm.libmodernsvm.kernelfunction;


/**
 * Radial basis function (RBF) kernel.
 */
public class RadialBasisKernelFunction<V extends Vector<V>> extends AbstractKernelFunction<V> {
  double gamma;

  public RadialBasisKernelFunction(double gamma) {
    super();
    this.gamma = gamma;
  }

  @Override
  public double similarity(V x, V y) {
    return Math.exp(-gamma * x.squareEuclidean(y));
  }

  public double gamma() {
    return gamma;
  }
}
