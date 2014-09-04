package com.kno10.svm.libmodernsvm;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

//
// Q matrices for various formulations
//
public class SVC_Q<T> extends KernelWithQD<T> {
	private final byte[] y;

	public SVC_Q(int l, T[] x_, KernelFunction<? super T> kf_, double cache_size,
			byte[] y_) {
		super(l, x_, kf_, cache_size);
		y = (byte[]) y_.clone();
	}

	@Override
	public float[] get_Q(int i, int len) {
		float[][] data = new float[1][];
		int start, j;
		if ((start = cache.get_data(i, data, len)) < len) {
			for (j = start; j < len; j++) {
				data[0][j] = (float) (y[i] * y[j] * kernel_function(i, j));
			}
		}
		return data[0];
	}

	@Override
	public void swap_index(int i, int j) {
		super.swap_index(i, j);
		byte tmpy = y[i];
		y[i] = y[j];
		y[j] = tmpy;
	}
}