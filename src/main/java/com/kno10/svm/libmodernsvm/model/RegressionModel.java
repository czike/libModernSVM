package com.kno10.svm.libmodernsvm.model;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public class RegressionModel<T> extends Model<T> {
	public double predict(T x, KernelFunction<? super T> kf, double[] dec_values) {
		double[] sv_coef = this.sv_coef[0];
		double sum = -rho[0];
		for (int i = 0; i < l; i++) {
			sum += sv_coef[i] * kf.similarity(x, SV[i]);
		}
		dec_values[0] = sum;
		// TODO: OneClass classification thresholds this value at 0.
		return sum;
	}

	public double predict(T x, KernelFunction<? super T> kf) {
		double[] sv_coef = this.sv_coef[0];
		double sum = -rho[0];
		for (int i = 0; i < l; i++) {
			sum += sv_coef[i] * kf.similarity(x, SV[i]);
		}
		// TODO: OneClass classification thresholds this value at 0.
		return sum;
	}
}
