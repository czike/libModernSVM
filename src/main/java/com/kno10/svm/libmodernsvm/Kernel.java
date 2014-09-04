package com.kno10.svm.libmodernsvm;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

abstract class Kernel implements QMatrix {
	private svm_node[][] x;

	public abstract float[] get_Q(int column, int len);

	public abstract double[] get_QD();

	public void swap_index(int i, int j) {
		svm_node[] tmp = x[i];
		x[i] = x[j];
		x[j] = tmp;
	}

	KernelFunction<svm_node[]> kf;

	double kernel_function(int i, int j) {
		return kf.kernel_function(x[i], x[j]);
	}

	Kernel(int l, svm_node[][] x_, KernelFunction<svm_node[]> kf_) {
		kf = kf_;
		x = (svm_node[][]) x_.clone();
	}
}