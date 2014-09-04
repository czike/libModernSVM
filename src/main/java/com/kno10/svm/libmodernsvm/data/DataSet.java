package com.kno10.svm.libmodernsvm.data;

/**
 * API to plug in custom data representations into libSVM.
 * 
 * @author Erich Schubert
 *
 * @param <T>
 *            Data type
 */
public interface DataSet<T> {
	/**
	 * Size of data set.
	 * 
	 * @return Data set size.
	 */
	int size();

	/**
	 * Get the ith element.
	 * 
	 * @param i
	 *            Element
	 * @return Element
	 */
	T get(int i);

	/**
	 * Get the value of the ith element.
	 * 
	 * @param i
	 *            Element
	 * @return Value
	 */
	double value(int i);

	/**
	 * Get the class number of the ith element.
	 * 
	 * @param i
	 *            Element
	 * @return Class number
	 */
	int classnum(int i);

	/**
	 * Swap two elements.
	 * 
	 * @param i
	 *            First position
	 * @param j
	 *            Second position
	 */
	void swap(int i, int j);

	/**
	 * Add a new element (optional operation).
	 * 
	 * @param v
	 *            New value
	 * @param weight
	 *            Weight value
	 */
	void add(T v, double weight);
	
	/**
	 * Reset the data set.
	 */
	void clear();
}