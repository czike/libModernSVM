package com.kno10.svm.libmodernsvm;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public abstract class KernelWithQD<T> extends Kernel<T> {
	private final double[] QD;

	public KernelWithQD(int l, T[] x_,
			KernelFunction<? super T> kf_, double cache_size) {
		super(l, x_, kf_, cache_size);

		QD = new double[l];
		for (int i = 0; i < l; i++)
			QD[i] = kernel_function(i, i);
	}

	@Override
	public void swap_index(int i, int j) {
		super.swap_index(i, j);
		// Swap QD values
		double tmpq = QD[i];
		QD[i] = QD[j];
		QD[j] = tmpq;
	}

	@Override
	public final double[] get_QD() {
		return QD;
	}
}
