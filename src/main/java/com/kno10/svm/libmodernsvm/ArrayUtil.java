package com.kno10.svm.libmodernsvm;

public class ArrayUtil {
	/**
	 * Swap two values in an object array.
	 * 
	 * @param data
	 *            Data
	 * @param i
	 *            First position
	 * @param j
	 *            Second position
	 * @param <T>
	 *            Object type
	 */
	public static <T> void swap(T[] data, int i, int j) {
		T tmp = data[i];
		data[i] = data[j];
		data[j] = tmp;
	}

	/**
	 * Swap two values in an integer array.
	 * 
	 * @param data
	 *            Data
	 * @param i
	 *            First position
	 * @param j
	 *            Second position
	 */
	public static void swap(int[] data, int i, int j) {
		int tmp = data[i];
		data[i] = data[j];
		data[j] = tmp;
	}

	/**
	 * Swap two values in a byte array.
	 * 
	 * @param data
	 *            Data
	 * @param i
	 *            First position
	 * @param j
	 *            Second position
	 */
	public static void swap(byte[] data, int i, int j) {
		byte tmp = data[i];
		data[i] = data[j];
		data[j] = tmp;
	}

	/**
	 * Swap two values in a double array.
	 * 
	 * @param data
	 *            Data
	 * @param i
	 *            First position
	 * @param j
	 *            Second position
	 */
	public static void swap(double[] data, int i, int j) {
		double tmp = data[i];
		data[i] = data[j];
		data[j] = tmp;
	}

	/**
	 * Swap two values in a float array.
	 * 
	 * @param data
	 *            Data
	 * @param i
	 *            First position
	 * @param j
	 *            Second position
	 */
	public static void swap(float[] data, int i, int j) {
		float tmp = data[i];
		data[i] = data[j];
		data[j] = tmp;
	}
}
