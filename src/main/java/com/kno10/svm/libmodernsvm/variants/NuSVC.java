package com.kno10.svm.libmodernsvm.variants;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelmatrix.SVC_Q;

public class NuSVC<T> extends AbstractSVC<T> {
	private static final Logger LOG = Logger.getLogger(NuSVC.class.getName());

	protected double nu;

	public NuSVC(double eps, boolean shrinking, double cache_size, double nu) {
		super(eps, shrinking, cache_size);
		this.nu = nu;
	}

	@Override
	protected Solver.SolutionInfo solve(DataSet<T> x,
			KernelFunction<? super T> kernel_function) {
		final int l = x.size();
		byte[] y = new byte[l];
		for (int i = 0; i < l; i++) {
			y[i] = (byte) ((x.value(i) > 0) ? +1 : -1);
		}

		double sum_pos = nu * l / 2, sum_neg = nu * l / 2;

		double[] alpha = new double[l];
		for (int i = 0; i < l; i++)
			if (y[i] == +1) {
				alpha[i] = Math.min(1.0, sum_pos);
				sum_pos -= alpha[i];
			} else {
				alpha[i] = Math.min(1.0, sum_neg);
				sum_neg -= alpha[i];
			}

		double[] zeros = new double[l];

		SVC_Q<T> Q = new SVC_Q<T>(x, kernel_function, cache_size, y);
		Solver.SolutionInfo si = new Solver_NU().solve(l, Q, zeros, y, alpha,
				1., 1., eps, shrinking);
		if (LOG.isLoggable(Level.INFO)) {
			LOG.info("C = " + 1 / si.r);
		}

		for (int i = 0; i < l; i++) {
			si.alpha[i] *= y[i] / si.r;
		}

		si.rho /= si.r;
		si.obj /= (si.r * si.r);
		si.upper_bound_p = 1 / si.r;
		si.upper_bound_n = 1 / si.r;
		return si;
	}

	@Override
	protected Logger getLogger() {
		return LOG;
	}
}