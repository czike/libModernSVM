package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public class SVR_Q<T> extends Kernel<T> {
	private final int l;
	private final byte[] sign;
	private final int[] index;
	private int next_buffer;
	private float[][] buffer;
	private final double[] QD;

	public SVR_Q(int l, T[] x_, KernelFunction<? super T> kf_, double cache_size) {
		super(l, x_, kf_, cache_size);
		this.l = l;
		QD = new double[2 * l];
		sign = new byte[2 * l];
		index = new int[2 * l];
		for (int k = 0; k < l; k++) {
			sign[k] = 1;
			sign[k + l] = -1;
			index[k] = k;
			index[k + l] = k;
			QD[k] = similarity(k, k);
			QD[k + l] = QD[k];
		}
		buffer = new float[2][2 * l];
		next_buffer = 0;
	}

	@Override
	public void swap_index(int i, int j) {
		// Swap sign
		byte tmps = sign[i];
		sign[i] = sign[j];
		sign[j] = tmps;
		// Swap indexes
		int tmpi = index[i];
		index[i] = index[j];
		index[j] = tmpi;
		// Swap QD
		double tmp = QD[i];
		QD[i] = QD[j];
		QD[j] = tmp;
	}

	@Override
	public float[] get_Q(int i, int len) {
		float[][] data = new float[1][];
		int j, real_i = index[i];
		data[0] = super.get_Q(real_i, l);

		// reorder and copy
		float buf[] = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		byte si = sign[i];
		for (j = 0; j < len; j++) {
			buf[j] = (float) si * sign[j] * data[0][index[j]];
		}
		return buf;
	}

	@Override
	public double[] get_QD() {
		return QD;
	}
}