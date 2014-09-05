package com.kno10.svm.libmodernsvm.kernelfunction.sparsevec;


/**
 * Linear kernel.
 */
public class LinearKernelFunction extends AbstractKernelFunction {
	public double similarity(SparseVector x, SparseVector y) {
		return dot(x, y);
	}
}
