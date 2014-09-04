package com.kno10.svm.libmodernsvm;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public abstract class Kernel<T> implements QMatrix {
	private Object[] x;

	protected final Cache cache;

	public abstract float[] get_Q(int column, int len);

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
}