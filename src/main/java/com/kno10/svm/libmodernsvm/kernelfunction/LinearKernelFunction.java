package com.kno10.svm.libmodernsvm.kernelfunction;

import com.kno10.svm.libmodernsvm.svm_node;

public class LinearKernelFunction extends AbstractKernelFunction {
	public double kernel_function(svm_node[] i, svm_node[] j) {
		return dot(i, j);
	}
}
