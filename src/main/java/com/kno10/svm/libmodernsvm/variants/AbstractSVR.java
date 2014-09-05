package com.kno10.svm.libmodernsvm.variants;

import java.util.ArrayList;
import java.util.logging.Level;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.data.DoubleWeightedArrayDataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.model.RegressionModel;

public abstract class AbstractSVR<T> extends AbstractSingleSVM<T> {
	public AbstractSVR(double eps, int shrinking, double cache_size) {
		super(eps, shrinking, cache_size);
	}

	public RegressionModel<T> make_model(DataSet<T> x) {
		final int l = x.size();
		// TODO: re-add probability support
		RegressionModel<T> model = new RegressionModel<T>();
		model.nr_class = 2;
		model.sv_coef = new double[1][];
		model.rho = new double[1];
		model.rho[0] = rho;

		int nSV = 0;
		for (int i = 0; i < l; i++) {
			if (Math.abs(alpha[i]) > 0) {
				++nSV;
			}
		}
		model.l = nSV;
		model.SV = new ArrayList<T>(nSV);
		model.sv_coef[0] = new double[nSV];
		model.sv_indices = new int[nSV];
		for (int i = 0, j = 0; i < l; i++) {
			if (Math.abs(alpha[i]) > 0) {
				model.SV.add(x.get(i));
				model.sv_coef[0][j] = alpha[i];
				model.sv_indices[j] = i + 1;
				++j;
			}
		}
		return model;
	}

	// Stratified cross validation
	public void svm_cross_validation_regression(DataSet<T> x,
			KernelFunction<? super T> kf, int nr_fold, double[] target) {
		final int l = x.size();
		int[] fold_start = new int[nr_fold + 1];
		int[] perm = shuffledIndex(new int[l], l);
		// Split into folds
		for (int i = 0; i <= nr_fold; i++) {
			fold_start[i] = i * l / nr_fold;
		}

		for (int i = 0; i < nr_fold; i++) {
			int begin = fold_start[i];
			int end = fold_start[i + 1];

			final int newl = l - (end - begin);
			DataSet<T> newx = new DoubleWeightedArrayDataSet<T>(newl);

			for (int j = 0; j < begin; ++j) {
				newx.add(x.get(perm[j]), x.value(perm[j]));
			}
			for (int j = end; j < l; ++j) {
				newx.add(x.get(perm[j]), x.value(perm[j]));
			}
			RegressionModel<T> submodel = svm_train_regression(newx, kf);
			for (int j = begin; j < end; j++) {
				target[perm[j]] = submodel.predict(x.get(perm[j]), kf);
			}
		}
	}

	private double svm_svr_probability(DataSet<T> x,
			KernelFunction<? super T> kf, double[] probA) {
		final int l = x.size();
		int nr_fold = 5;

		double[] ymv = new double[l];
		double mae = 0;
		svm_cross_validation_regression(x, kf, nr_fold, ymv);
		for (int i = 0; i < l; i++) {
			ymv[i] = x.value(i) - ymv[i];
			mae += Math.abs(ymv[i]);
		}
		mae /= l;
		double std = Math.sqrt(2) * mae;
		int count = 0;
		mae = 0;
		for (int i = 0; i < l; i++) {
			if (Math.abs(ymv[i]) > 5 * std) {
				++count;
			} else {
				mae += Math.abs(ymv[i]);
			}
		}
		mae /= (l - count);
		if (getLogger().isLoggable(Level.INFO)) {
			getLogger()
					.info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma="
							+ mae + "\n");
		}
		return mae;
	}

	boolean probability = false;

	public RegressionModel<T> svm_train_regression(DataSet<T> x,
			KernelFunction<? super T> kf) {
		// FIXME: Probability support is incomplete.
		if (probability) {
			double[] probA = new double[1];
			probA[0] = svm_svr_probability(x, kf, probA);
		}

		svm_train_one(x, kf);
		return make_model(x);
	}
}
