package com.kno10.svm.libmodernsvm.variants;

import java.util.Arrays;

import com.kno10.svm.libmodernsvm.model.RegressionModel;

public abstract class AbstractSVR<T> extends AbstractSingleSVM<T> {
	public AbstractSVR(double eps, int shrinking, double cache_size) {
		super(eps, shrinking, cache_size);
	}

	public RegressionModel<T> make_model(int l, T[] x) {
		// TODO: re-add probability support
		RegressionModel<T> model = new RegressionModel<T>();
		model.nr_class = 2;
		model.sv_coef = new double[1][];
		model.rho = new double[1];
		model.rho[0] = rho;

		int nSV = 0;
		for (int i = 0; i < l; i++)
			if (Math.abs(alpha[i]) > 0)
				++nSV;
		model.l = nSV;
		model.SV = Arrays.copyOf(x, nSV); // FIXME: this is a hack.
		model.sv_coef[0] = new double[nSV];
		model.sv_indices = new int[nSV];
		for (int i = 0, j = 0; i < l; i++)
			if (Math.abs(alpha[i]) > 0) {
				model.SV[j] = x[i];
				model.sv_coef[0][j] = alpha[i];
				model.sv_indices[j] = i + 1;
				++j;
			}
		return model;
	}
}
