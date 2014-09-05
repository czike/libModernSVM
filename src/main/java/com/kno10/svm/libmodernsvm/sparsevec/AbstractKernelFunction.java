package com.kno10.svm.libmodernsvm.sparsevec;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public abstract class AbstractKernelFunction implements
		KernelFunction<SparseVector> {
	protected static double dot(SparseVector x, SparseVector y) {
		double sum = 0.;
		final int xlen = x.index.length, ylen = y.index.length;
		int i = 0, j = 0;
		while (i < xlen && j < ylen) {
			int xi = x.index[i], yi = y.index[j];
			if (xi == yi) {
				sum += x.value[i++] * y.value[j++];
			} else {
				if (xi > yi) {
					++j;
				} else {
					++i;
				}
			}
		}
		return sum;
	}

	protected static double powi(double base, int times) {
		double tmp = base, ret = 1.;
		for (int t = times; t > 0; t >>>= 2) {
			if ((t & 0x1) == 1) {
				ret *= tmp;
			}
			tmp *= tmp;
		}
		return ret;
	}
}
