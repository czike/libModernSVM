package com.kno10.svm.libmodernsvm.kernelfunction;

import com.kno10.svm.libmodernsvm.svm_node;

public class PolynomialKernelFunction extends AbstractKernelFunction {
	private final int degree;
	private final double gamma;
	private final double coef0;

	public PolynomialKernelFunction(int degree, double gamma, double coef0) {
		super();
		this.degree = degree;
		this.gamma = gamma;
		this.coef0 = coef0;
	}

	public double kernel_function(svm_node[] i, svm_node[] j) {
		return powi(gamma * dot(i, j) + coef0, degree);
	}
}
