package com.kno10.svm.libmodernsvm.kernelfunction;

import com.kno10.svm.libmodernsvm.svm_node;

/**
 * Polynomial kernel.
 */
public class PolynomialKernelFunction extends AbstractKernelFunction {
	/** Kernel degree */
	private final int degree;
	/** Gamma factor */
	private final double gamma;
	/** Offset coefficient */
	private final double coef0;

	public PolynomialKernelFunction(int degree, double gamma, double coef0) {
		super();
		this.degree = degree;
		this.gamma = gamma;
		this.coef0 = coef0;
	}

	public double similarity(svm_node[] x, svm_node[] y) {
		return powi(gamma * dot(x, y) + coef0, degree);
	}
}
