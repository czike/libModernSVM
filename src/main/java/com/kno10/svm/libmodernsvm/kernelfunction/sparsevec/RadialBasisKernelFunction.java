package com.kno10.svm.libmodernsvm.kernelfunction.sparsevec;

/**
 * Radial basis function (RBF) kernel.
 */
public class RadialBasisKernelFunction extends AbstractKernelFunction {
	double gamma;

	public RadialBasisKernelFunction(double gamma) {
		super();
		this.gamma = gamma;
	}

	public double similarity(SparseVector x, SparseVector y) {
		double sum = 0;
		final int xlen = x.index.length, ylen = y.index.length;
		int i = 0, j = 0;
		while (i < xlen && j < ylen) {
			if (x.index[i] == y.index[j]) {
				double d = x.value[i++] - y.value[j++];
				sum += d * d;
			} else if (x.index[i] > y.index[j]) {
				sum += y.value[j] * y.value[j];
				++j;
			} else {
				sum += x.value[i] * x.value[i];
				++i;
			}
		}

		while (i < xlen) {
			sum += x.value[i] * x.value[i];
			++i;
		}

		while (j < ylen) {
			sum += y.value[j] * y.value[j];
			++j;
		}

		return Math.exp(-gamma * sum);
	}
}
