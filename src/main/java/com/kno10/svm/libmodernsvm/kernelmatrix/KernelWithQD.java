package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.ArrayUtil;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public abstract class KernelWithQD<T> extends Kernel<T> {
	// Diagonal values <x,x>
	private final double[] QD;

	public KernelWithQD(DataSet<T> x, KernelFunction<? super T> kf,
			double cache_size) {
		super(x, kf, cache_size);
		final int l = x.size();
		QD = new double[l];
		for (int i = 0; i < l; i++) {
			QD[i] = kernel_function(i, i);
		}
	}

	@Override
	public void swap_index(int i, int j) {
		super.swap_index(i, j);
		ArrayUtil.swap(QD, i, j);
	}

	@Override
	public final double[] get_QD() {
		return QD;
	}
}
