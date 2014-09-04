package com.kno10.svm.libmodernsvm.variants;

public abstract class AbstractSVC<T> extends AbstractSingleSVM<T> {

	public AbstractSVC(double eps, int shrinking, double cache_size) {
		super(eps, shrinking, cache_size);
	}
}
