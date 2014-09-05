package com.kno10.svm.libmodernsvm.kernelfunction.libsvm;

/**
 * Sparse data model used by the original libSVM code (svm_node)
 */
public class SparseVectorEntry {
	public int index;
	public double value;

	public SparseVectorEntry(int index, double value) {
		super();
		this.index = index;
		this.value = value;
	}
}
