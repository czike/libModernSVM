package com.kno10.svm.libmodernsvm.kernelfunction.unsafe;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public abstract class AbstractKernelFunction implements KernelFunction<UnsafeSparseVector> {
  protected static double powi(double base, int exp) {
    if(exp <= 2) {
      return Math.pow(base, exp);
    }
    double tmp = base, ret = (exp & 1) == 1 ? base : 1.;
    exp >>= 1;
    while(true) {
      if(exp == 1) {
        return ret * tmp;
      }
      if((exp & 1) != 0) {
        ret *= tmp;
      }
      tmp *= tmp;
      exp >>= 1;
    }
  }
}
