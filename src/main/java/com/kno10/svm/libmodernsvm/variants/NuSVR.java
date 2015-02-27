package com.kno10.svm.libmodernsvm.variants;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelmatrix.SVR_Q;

public class NuSVR<T> extends AbstractSVR<T> {
  private static final Logger LOG = Logger.getLogger(NuSVR.class.getName());

  protected double nu, C;

  public NuSVR(double eps, boolean shrinking, double cache_size, double C, double nu) {
    super(eps, shrinking, cache_size);
    this.nu = nu;
    this.C = C;
  }

  @Override
  protected Solver.SolutionInfo solve(DataSet<T> x, KernelFunction<? super T> kernel_function) {
    final int l = x.size();
    double[] alpha2 = new double[2 * l];
    double[] linear_term = new double[2 * l];
    byte[] y = new byte[2 * l];

    double sum = C * nu * l / 2;
    for(int i = 0; i < l; i++) {
      alpha2[i] = alpha2[i + l] = Math.min(sum, C);
      sum -= alpha2[i];

      linear_term[i] = -y[i];
      y[i] = 1;

      linear_term[i + l] = y[i];
      y[i + l] = -1;
    }

    SVR_Q Q = new SVR_Q(x, kernel_function, cache_size);
    Solver_NU solver = new Solver_NU();
    Solver.SolutionInfo si = solver.solve(2 * l, Q, linear_term, y, alpha2, C, C, eps, shrinking);

    if(LOG.isLoggable(Level.INFO)) {
      LOG.info("epsilon = " + (-solver.r));
    }

    for(int i = 0; i < l; i++) {
      si.alpha[i] = alpha2[i] - alpha2[i + l];
    }
    return si;
  }

  @Override
  protected Logger getLogger() {
    return LOG;
  }
}