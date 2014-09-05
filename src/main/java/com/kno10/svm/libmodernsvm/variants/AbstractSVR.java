package com.kno10.svm.libmodernsvm.variants;

import java.util.ArrayList;
import java.util.logging.Level;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.data.DoubleWeightedArrayDataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.model.ProbabilisticRegressionModel;
import com.kno10.svm.libmodernsvm.model.RegressionModel;

public abstract class AbstractSVR<T> extends AbstractSingleSVM<T> {
	public AbstractSVR(double eps, int shrinking, double cache_size) {
		super(eps, shrinking, cache_size);
	}

	public RegressionModel<T> make_model(Solver.SolutionInfo si, DataSet<T> x,
			double[] probA) {
		final int l = x.size();
		// TODO: re-add probability support
		RegressionModel<T> model;
		if (probA == null) {
			model = new RegressionModel<T>();
		} else {
			ProbabilisticRegressionModel<T> pm = new ProbabilisticRegressionModel<T>();
			pm.probA = probA;
			model = pm;
		}
		model.nr_class = 2;
		model.sv_coef = new double[1][];
		model.rho = new double[1];
		model.rho[0] = si.rho;

		int nSV = 0;
		for (int i = 0; i < l; i++) {
			if (nonzero(si.alpha[i])) {
				++nSV;
			}
		}
		model.l = nSV;
		model.SV = new ArrayList<T>(nSV);
		model.sv_coef[0] = new double[nSV];
		model.sv_indices = new int[nSV];
		for (int i = 0, j = 0; i < l; i++) {
			if (nonzero(si.alpha[i])) {
				model.SV.add(x.get(i));
				model.sv_coef[0][j] = si.alpha[i];
				model.sv_indices[j] = i + 1;
				++j;
			}
		}
		return model;
	}

	// Perform cross-validation.
	public void cross_validation(DataSet<T> x, KernelFunction<? super T> kf,
			int nr_fold, double[] target) {
		final int l = x.size();
		int[] perm = shuffledIndex(new int[l], l);
		// Split into folds
		int[] fold_start = makeFolds(l, nr_fold);

		DataSet<T> newx = new DoubleWeightedArrayDataSet<T>(l);
		for (int i = 0; i < nr_fold; i++) {
			final int begin = fold_start[i], end = fold_start[i + 1];

			newx.clear();
			for (int j = 0; j < begin; ++j) {
				newx.add(x.get(perm[j]), x.value(perm[j]));
			}
			for (int j = end; j < l; ++j) {
				newx.add(x.get(perm[j]), x.value(perm[j]));
			}
			RegressionModel<T> submodel = train(newx, kf);
			for (int j = begin; j < end; j++) {
				target[perm[j]] = submodel.predict(x.get(perm[j]), kf);
			}
		}
	}

	private double svr_probability(DataSet<T> x, KernelFunction<? super T> kf,
			double[] probA) {
		final int l = x.size();
		int nr_fold = 5;

		double[] ymv = new double[l];
		double mae = 0;
		cross_validation(x, kf, nr_fold, ymv);
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
					.info("Prob. model for test data: target value = predicted value + z,\n"
							+ "z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma="
							+ mae);
		}
		return mae;
	}

	boolean probability = false;

	public RegressionModel<T> train(DataSet<T> x, KernelFunction<? super T> kf) {
		double[] probA = null;
		if (probability) {
			probA = new double[1];
			probA[0] = svr_probability(x, kf, probA);
		}

		Solver.SolutionInfo si = train_one(x, kf);
		return make_model(si, x, probA);
	}
}
