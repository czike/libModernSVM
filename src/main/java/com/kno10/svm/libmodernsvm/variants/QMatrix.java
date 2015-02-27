package com.kno10.svm.libmodernsvm.variants;

/**
 * API to get kernel similarity values.
 */
public interface QMatrix {
	/**
	 * Get a column of the matrix.
	 * 
	 * @param column Column number
	 * @param len Number of entries to get
	 * @param out Output array for similarity values
	 */
	void get_Q(int column, int len, float[] out);

	/**
	 * Get the diagonal values, as reference.
	 * 
	 * @return Diagonal values
	 */
	double[] get_QD();

	/**
	 * Reorganize the data by swapping two entries.
	 * 
	 * This also must modify the QD array!
	 * 
	 * @param i First entry
	 * @param j Second entry
	 */
	void swap_index(int i, int j);

	/**
	 * Distance, with sign parameter.
	 * 
	 * @param i First entry
	 * @param j Second entry
	 * @param b Sign
	 * @return Distance
	 */
  double quadDistance(int i, int j, byte b);
}