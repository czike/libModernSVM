package com.kno10.svm.libmodernsvm.kernelfunction;

import com.kno10.svm.libmodernsvm.svm_node;

public class RadialBasisKernelFunction extends AbstractKernelFunction {
	double gamma;

	public RadialBasisKernelFunction(double gamma) {
		super();
		this.gamma = gamma;
	}

	public double kernel_function(svm_node[] x, svm_node[] y) {
		double sum = 0;
		final int xlen = x.length, ylen = y.length;
		int i = 0, j = 0;
		while (i < xlen && j < ylen) {
			if (x[i].index == y[j].index) {
				double d = x[i++].value - y[j++].value;
				sum += d * d;
			} else if (x[i].index > y[j].index) {
				sum += y[j].value * y[j].value;
				++j;
			} else {
				sum += x[i].value * x[i].value;
				++i;
			}
		}

		while (i < xlen) {
			sum += x[i].value * x[i].value;
			++i;
		}

		while (j < ylen) {
			sum += y[j].value * y[j].value;
			++j;
		}

		return Math.exp(-gamma * sum);
	}
}
