package com.kno10.svm.libmodernsvm.variants;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelmatrix.SVC_Q;

public class SVM_C<T> extends AbstractSingleSVM<T> {
	private static final Logger LOG = Logger.getLogger(SVM_C.class.getName());

	double Cp, Cn;

	public SVM_C(double eps, int shrinking, double cache_size,double Cp, double Cn) {
		super(eps, shrinking, cache_size);
		this.Cp = Cp;
		this.Cn = Cn;
	}

	@Override
	protected Solver.SolutionInfo solve(int l, T[] x, double[] y_, KernelFunction<? super T> kernel_function) {
		double[] minus_ones = new double[l];
		byte[] y = new byte[l];

		for (int i = 0; i < l; i++) {
			alpha[i] = 0;
			minus_ones[i] = -1;
			y[i] = (byte) ((y_[i] > 0) ? +1 : -1);
		}

		Solver.SolutionInfo si = new Solver().solve(l, new SVC_Q<T>(l, x, kernel_function, cache_size, y),
				minus_ones, y, alpha, Cp, Cn, eps, shrinking);

		if (Cp == Cn && LOG.isLoggable(Level.INFO)) {
			double sum_alpha = 0;
			for (int i = 0; i < l; i++) {
				sum_alpha += alpha[i];
			}

			LOG.info("nu = " + sum_alpha / (Cp * l) + "\n");
		}

		for (int i = 0; i < l; i++) {
			alpha[i] *= y[i];
		}
		return si;
	}

	@Override
	protected Logger getLogger() {
		return LOG;
	}
}