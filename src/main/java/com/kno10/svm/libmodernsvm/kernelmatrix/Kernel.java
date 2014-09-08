package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.variants.QMatrix;

public abstract class Kernel<T> implements QMatrix {
	protected final Cache<T> cache;

	abstract public double[] get_QD();

	public void swap_index(int i, int j) {
		// Swap in cache, too:
		cache.swap_index(i, j);
	}

	public Kernel(DataSet<T> x, KernelFunction<? super T> kf, double cache_size) {
	  super();
		cache = new Cache<T>(x, kf, (long) (cache_size * (1 << 20)));
	}

	public void get_Q(int i, int len, float[] out) {
		float[][] data = new float[1][];
		int start;
		if ((start = cache.get_data(i, data, len)) < len) {
			for (int j = start; j < len; j++) {
				data[0][j] = (float) similarity(i, j);
			}
		}
		System.arraycopy(data[0], 0, out, 0, len);
	}

	// Uncached similarity.
	public double similarity(int i, int j) {
		return cache.similarity(i, j);
	}
}