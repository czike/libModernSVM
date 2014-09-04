package com.kno10.svm.libmodernsvm;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public class ONE_CLASS_Q<T> extends KernelWithQD<T> {
	public ONE_CLASS_Q(int l, T[] x_, KernelFunction<? super T> kf_,
			double cache_size) {
		super(l, x_, kf_, cache_size);
	}

	@Override
	public float[] get_Q(int i, int len) {
		float[][] data = new float[1][];
		int start, j;
		if ((start = cache.get_data(i, data, len)) < len) {
			for (j = start; j < len; j++) {
				data[0][j] = (float) kernel_function(i, j);
			}
		}
		return data[0];
	}
}