package com.kno10.svm.libmodernsvm.variants;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelmatrix.SVR_Q;

public class SVR_Epsilon<T> extends AbstractSVR<T> {
	private static final Logger LOG = Logger.getLogger(SVR_Epsilon.class
			.getName());
	protected double p, C;

	public SVR_Epsilon(double eps, int shrinking, double cache_size, double C,
			double p) {
		super(eps, shrinking, cache_size);
		this.p = p;
		this.C = C;
	}

	@Override
	protected Solver.SolutionInfo solve(DataSet<T> x,
			KernelFunction<? super T> kernel_function) {
		final int l = x.size();
		double[] alpha2 = new double[2 * l];
		double[] linear_term = new double[2 * l];
		byte[] y = new byte[2 * l];

		for (int i = 0; i < l; i++) {
			alpha2[i] = 0;
			linear_term[i] = p - x.value(i);
			y[i] = 1;

			alpha2[i + l] = 0;
			linear_term[i + l] = p + x.value(i);
			y[i + l] = -1;
		}

		SVR_Q<T> Q = new SVR_Q<T>(x, kernel_function, cache_size);
		Solver.SolutionInfo si = new Solver().solve(2 * l, Q, linear_term, y,
				alpha2, C, C, eps, shrinking);

		if (LOG.isLoggable(Level.INFO)) {
			double sum_alpha = 0;
			for (int i = 0; i < l; i++) {
				alpha[i] = alpha2[i] - alpha2[i + l];
				sum_alpha += Math.abs(alpha[i]);
			}
			LOG.info("nu = " + sum_alpha / (C * l) + "\n");
		}
		return si;
	}

	@Override
	protected Logger getLogger() {
		return LOG;
	}
}