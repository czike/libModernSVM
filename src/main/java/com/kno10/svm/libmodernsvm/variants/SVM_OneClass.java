package com.kno10.svm.libmodernsvm.variants;

import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelmatrix.ONE_CLASS_Q;

public class SVM_OneClass<T> extends AbstractSingleSVM<T> {
	private static final Logger LOG = Logger.getLogger(SVM_OneClass.class
			.getName());
	protected double nu;

	public SVM_OneClass(double eps, int shrinking, double cache_size,
			KernelFunction<? super T> kernel_function, double nu) {
		super(eps, shrinking, cache_size, kernel_function);
		this.nu = nu;
	}

	@Override
	protected Solver.SolutionInfo solve(int l, T[] x, double[] y) {
		double[] zeros = new double[l];
		byte[] ones = new byte[l];

		int n = (int) (nu * l); // # of alpha's at upper bound

		for (int i = 0; i < n; i++) {
			alpha[i] = 1;
		}
		if (n < l) {
			alpha[n] = nu * l - n;
		}
		for (int i = n + 1; i < l; i++) {
			alpha[i] = 0;
		}

		for (int i = 0; i < l; i++) {
			zeros[i] = 0;
			ones[i] = 1;
		}

		return new Solver().solve(l, new ONE_CLASS_Q<T>(l, x, kernel_function,
				cache_size), zeros, ones, alpha, 1.0, 1.0, eps, shrinking);
	}

	@Override
	protected Logger getLogger() {
		return LOG;
	}
}