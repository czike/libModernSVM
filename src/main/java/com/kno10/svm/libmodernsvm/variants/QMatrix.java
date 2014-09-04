package com.kno10.svm.libmodernsvm.variants;

/**
 * API to get kernel similarity values.
 */
public interface QMatrix {
	float[] get_Q(int column, int len);

	double[] get_QD();

	/**
	 * Reorganize the data by swapping two entries.
	 * 
	 * @param i First entry
	 * @param j Second entry
	 */
	void swap_index(int i, int j);
}