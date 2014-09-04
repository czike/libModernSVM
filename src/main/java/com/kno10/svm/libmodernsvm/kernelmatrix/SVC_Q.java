package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

//
// Q matrices for various formulations
//
public class SVC_Q<T> extends KernelWithQD<T> {
	private final byte[] y;

	public SVC_Q(int l, T[] x_, KernelFunction<? super T> kf_,
			double cache_size, byte[] y_) {
		super(l, x_, kf_, cache_size);
		y = (byte[]) y_.clone();
	}

	@Override
	public double similarity(int i, int j) {
		return y[i] * y[j] * kernel_function(i, j);
	}

	@Override
	public void swap_index(int i, int j) {
		super.swap_index(i, j);
		byte tmpy = y[i];
		y[i] = y[j];
		y[j] = tmpy;
	}
}