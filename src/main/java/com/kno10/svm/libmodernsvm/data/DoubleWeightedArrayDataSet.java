package com.kno10.svm.libmodernsvm.data;

import java.util.Arrays;

/**
 * This is an efficient array based data set implementation.
 * 
 * @author Erich Schubert
 *
 * @param <T>
 *            Object type
 */
public class DoubleWeightedArrayDataSet<T> implements DataSet<T> {
	Object[] data;
	double[] weight;
	int size = 0;

	public DoubleWeightedArrayDataSet(int size) {
		data = new Object[size];
		weight = new double[size];
	}

	public int size() {
		return size;
	}

	@SuppressWarnings("unchecked")
	public T get(int i) {
		return (T) data[i];
	}

	public double value(int i) {
		return weight[i];
	}

	public int classnum(int i) {
		// Probably not what you meant to do!
		return (int) weight[i];
	}

	public void swap(int i, int j) {
		Object tmp = data[i];
		data[i] = data[j];
		data[j] = tmp;
		double dt = weight[i];
		weight[i] = weight[j];
		weight[j] = dt;
	}

	public void add(T v, double w) {
		if (size == data.length) {
			final int newlen = data.length << 1;
			data = Arrays.copyOf(data, newlen);
			weight = Arrays.copyOf(weight, newlen);
		}
		data[size] = v;
		weight[size] = w;
		size++;
	}
}
