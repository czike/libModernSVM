package com.kno10.svm.libmodernsvm.kernelfunction;

import com.kno10.svm.libmodernsvm.svm_node;

public class SigmoidKernelFunction extends AbstractKernelFunction {
	private final double gamma;
	private final double coef0;

	public SigmoidKernelFunction(double gamma, double coef0) {
		super();
		this.gamma = gamma;
		this.coef0 = coef0;
	}

	public double kernel_function(svm_node[] i, svm_node[] j) {
		return Math.tanh(gamma*dot(i,j)+coef0);
	}
}
