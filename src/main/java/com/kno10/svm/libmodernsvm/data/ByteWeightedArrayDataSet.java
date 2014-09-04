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
public class ByteWeightedArrayDataSet<T> implements DataSet<T> {
	Object[] data;
	byte[] weight;
	int size = 0;

	public ByteWeightedArrayDataSet(int size) {
		data = new Object[size];
		weight = new byte[size];
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
		byte dt = weight[i];
		weight[i] = weight[j];
		weight[j] = dt;
	}

	public void add(T v, double w) {
		add(v, (byte) w);
	}

	public void add(T v, byte w) {
		if (size == data.length) {
			final int newlen = data.length << 1;
			data = Arrays.copyOf(data, newlen);
			weight = Arrays.copyOf(weight, newlen);
		}
		data[size] = v;
		weight[size] = w;
		size++;
	}

	public void clear() {
		size = 0;
	}
}
