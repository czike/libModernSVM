package com.kno10.svm.libmodernsvm.kernelfunction.libsvm;


/**
 * Linear kernel.
 */
public class LinearKernelFunction extends AbstractKernelFunction {
	public double similarity(SparseVectorEntry[] x, SparseVectorEntry[] y) {
		return dot(x, y);
	}
}
