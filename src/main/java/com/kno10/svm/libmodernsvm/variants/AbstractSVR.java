package com.kno10.svm.libmodernsvm.variants;

public abstract class AbstractSVR<T> extends AbstractSingleSVM<T> {
	public AbstractSVR(double eps, int shrinking, double cache_size) {
		super(eps, shrinking, cache_size);
	}
}
