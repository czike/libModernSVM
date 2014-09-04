package com.kno10.svm.libmodernsvm.variants;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelmatrix.SVC_Q;

/**
 * Regularized SVM based classification (C-SVC).
 *
 * @param <T>
 */
public class SVC_C<T> extends AbstractSVC<T> {
	private static final Logger LOG = Logger.getLogger(SVC_C.class.getName());

	double Cp = 1., Cn = 1.;

	public SVC_C(double eps, int shrinking, double cache_size) {
		super(eps, shrinking, cache_size);
	}

	@Override
	public void set_weights(double Cp, double Cn) {
		this.Cp = Cp;
		this.Cn = Cn;
	}

	@Override
	protected Solver.SolutionInfo solve(DataSet<T> x,
			KernelFunction<? super T> kernel_function) {
		final int l = x.size();
		double[] minus_ones = new double[l];
		byte[] y = new byte[l];

		for (int i = 0; i < l; i++) {
			alpha[i] = 0;
			minus_ones[i] = -1;
			y[i] = (byte) ((x.value(i) > 0) ? +1 : -1);
		}

		Solver.SolutionInfo si = new Solver().solve(l, new SVC_Q<T>(x,
				kernel_function, cache_size, y), minus_ones, y, alpha, Cp, Cn,
				eps, shrinking);

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