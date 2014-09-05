package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.ArrayUtil;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.variants.CSVC;
import com.kno10.svm.libmodernsvm.variants.NuSVC;

/**
 * Q matrix used by {@link CSVC} and {@link NuSVC} classification.
 *
 * @param <T>
 */
public class SVC_Q<T> extends KernelWithQD<T> {
	private final byte[] y;

	public SVC_Q(DataSet<T> x, KernelFunction<? super T> kf, double cache_size,
			byte[] y) {
		super(x, kf, cache_size);
		this.y = (byte[]) y.clone();
	}

	@Override
	public double similarity(int i, int j) {
		return y[i] * y[j] * kernel_function(i, j);
	}

	@Override
	public void swap_index(int i, int j) {
		super.swap_index(i, j);
		ArrayUtil.swap(y, i, j);
	}
}