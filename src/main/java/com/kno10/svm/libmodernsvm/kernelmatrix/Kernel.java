package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.QMatrix;
import com.kno10.svm.libmodernsvm.svm_node;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public abstract class Kernel<T> implements QMatrix {
	private Object[] x;

	private final Cache cache;

	abstract public double[] get_QD();

	public void swap_index(int i, int j) {
		// Swap nodes
		Object tmp = x[i];
		x[i] = x[j];
		x[j] = tmp;
		// Swap in cache, too:
		cache.swap_index(i, j);
	}

	KernelFunction<? super T> kf;

	@SuppressWarnings("unchecked")
	double kernel_function(int i, int j) {
		return kf.kernel_function((T) x[i], (T) x[j]);
	}

	public Kernel(int l, T[] x_, KernelFunction<? super T> kf_,
			double cache_size) {
		kf = kf_;
		x = (svm_node[][]) x_.clone();
		cache = new Cache(l, (long) (cache_size * (1 << 20)));
	}

	public float[] get_Q(int i, int len) {
		float[][] data = new float[1][];
		int start, j;
		if ((start = cache.get_data(i, data, len)) < len) {
			for (j = start; j < len; j++) {
				data[0][j] = (float) similarity(i, j);
			}
		}
		return data[0];
	}

	// Uncached similarity.
	public double similarity(int i, int j) {
		return kernel_function(i, j);
	}
}