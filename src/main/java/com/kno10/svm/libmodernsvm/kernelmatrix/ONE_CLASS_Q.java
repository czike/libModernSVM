package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.variants.SVR_OneClass;

/**
 * Q matrix used by one-class classification {@link SVR_OneClass}.
 * 
 * Similar to {@link SVC_Q}, but by definition all training data is positive.
 *
 * @param <T>
 */
public class ONE_CLASS_Q<T> extends KernelWithQD<T> {
	public ONE_CLASS_Q(int l, T[] x_, KernelFunction<? super T> kf_,
			double cache_size) {
		super(l, x_, kf_, cache_size);
	}
}