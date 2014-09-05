package com.kno10.svm.libmodernsvm.variants;

import java.util.Arrays;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelmatrix.ONE_CLASS_Q;

/**
 * One-class classification is similar to regression.
 *
 * @param <T>
 */
public class OneClassSVM<T> extends AbstractSVR<T> {
	private static final Logger LOG = Logger.getLogger(OneClassSVM.class
			.getName());
	protected double nu;

	public OneClassSVM(double eps, int shrinking, double cache_size, double nu) {
		super(eps, shrinking, cache_size);
		this.nu = nu;
	}

	@Override
	protected Solver.SolutionInfo solve(DataSet<T> x,
			KernelFunction<? super T> kernel_function) {
		final int l = x.size();
		double[] zeros = new double[l];
		byte[] ones = new byte[l];

		final int n = (int) (nu * l); // # of alpha's at upper bound

		double[] alpha = new double[l];
		for (int i = 0; i < n; i++) {
			alpha[i] = 1;
		}
		if (n < l) {
			alpha[n] = nu * l - n;
		}
		for (int i = n + 1; i < l; i++) {
			alpha[i] = 0;
		}
		Arrays.fill(ones, (byte) 1);

		ONE_CLASS_Q<T> Q = new ONE_CLASS_Q<T>(x, kernel_function, cache_size);
		return new Solver().solve(l, Q, zeros, ones, alpha, 1.0, 1.0, eps,
				shrinking);
	}

	@Override
	protected Logger getLogger() {
		return LOG;
	}
}