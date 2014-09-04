package com.kno10.svm.libmodernsvm.variants;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public abstract class AbstractSingleSVM<T> {
	protected double eps;
	protected int shrinking;
	protected double cache_size;
	protected KernelFunction<? super T> kernel_function;

	// Output variables
	public double[] alpha;
	public double rho;

	public AbstractSingleSVM(double eps, int shrinking, double cache_size,
			KernelFunction<? super T> kernel_function) {
		this.eps = eps;
		this.shrinking = shrinking;
		this.cache_size = cache_size;
		this.kernel_function = kernel_function;
	}

	abstract protected Solver.SolutionInfo solve(int l, T[] x, double[] y);

	public void svm_train_one(int l, T[] x, double[] y) {
		alpha = new double[l];
		Solver.SolutionInfo si = solve(l, x, y);
		rho = si.rho;

		if (getLogger().isLoggable(Level.INFO)) {
			getLogger().info("obj = " + si.obj + ", rho = " + si.rho + "\n");
		}

		// output SVs

		int nSV = 0;
		int nBSV = 0;
		for (int i = 0; i < l; i++) {
			if (Math.abs(alpha[i]) > 0) {
				++nSV;
				if (Math.abs(alpha[i]) >= ((y[i] > 0) ? si.upper_bound_p
						: si.upper_bound_n))
					++nBSV;
			}
		}

		if (getLogger().isLoggable(Level.INFO)) {
			getLogger().info("nSV = " + nSV + ", nBSV = " + nBSV + "\n");
		}
	}

	abstract protected Logger getLogger();
}