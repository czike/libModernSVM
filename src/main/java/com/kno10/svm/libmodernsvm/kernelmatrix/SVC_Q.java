package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.ArrayUtil;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.variants.SVC_C;
import com.kno10.svm.libmodernsvm.variants.SVC_Nu;

/**
 * Q matrix used by {@link SVC_C} and {@link SVC_Nu} classification.
 *
 * @param <T>
 */
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
		ArrayUtil.swap(y, i, j);
	}
}