package com.kno10.svm.libmodernsvm.sparsevec;

/**
 * More compact sparse vector type.
 */
public class SparseVector {
	public int[] index;
	public double[] value;

	public SparseVector(int[] index, double[] value) {
		super();
		this.index = index;
		this.value = value;
	}
}
