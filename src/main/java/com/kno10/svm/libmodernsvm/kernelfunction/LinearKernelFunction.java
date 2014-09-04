package com.kno10.svm.libmodernsvm.kernelfunction;

import com.kno10.svm.libmodernsvm.svm_node;

/**
 * Linear kernel.
 */
public class LinearKernelFunction extends AbstractKernelFunction {
	public double similarity(svm_node[] x, svm_node[] y) {
		return dot(x, y);
	}
}
