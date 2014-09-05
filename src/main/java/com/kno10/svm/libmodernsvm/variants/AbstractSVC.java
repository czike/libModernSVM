package com.kno10.svm.libmodernsvm.variants;

import java.util.ArrayList;

import com.kno10.svm.libmodernsvm.data.ByteWeightedArrayDataSet;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.data.DoubleWeightedArrayDataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.model.ClassificationModel;
import com.kno10.svm.libmodernsvm.model.ProbabilisticClassificationModel;

public abstract class AbstractSVC<T> extends AbstractSingleSVM<T> {

	public AbstractSVC(double eps, int shrinking, double cache_size) {
		super(eps, shrinking, cache_size);
	}

	boolean probability = false;

	public ClassificationModel<T> svm_train_classification(DataSet<T> x,
			KernelFunction<? super T> kf, double[] weighted_C) {
		final int l = x.size();
		// classification
		int[] tmp_nr_class = new int[1];
		int[][] tmp_label = new int[1][];
		int[][] tmp_start = new int[1][];
		int[][] tmp_count = new int[1][];
		int[] perm = new int[l];

		// group training data of the same class
		svm_group_classes(x, tmp_nr_class, tmp_label, tmp_start, tmp_count,
				perm);
		int nr_class = tmp_nr_class[0];
		int[] label = tmp_label[0];
		int[] start = tmp_start[0];
		int[] count = tmp_count[0];

		if (nr_class == 1)
			getLogger()
					.info("WARNING: training data in only one class. See README for details.\n");

		// train k*(k-1)/2 binary models

		boolean[] nonzero = new boolean[l];
		for (int i = 0; i < l; i++)
			nonzero[i] = false;
		final int pairs = nr_class * (nr_class - 1) / 2;
		double[][] f_alpha = new double[pairs][];
		double[] f_rho = new double[pairs];

		double[] probA = probability ? new double[pairs] : null;
		double[] probB = probability ? new double[pairs] : null;

		DataSet<T> newx = new ByteWeightedArrayDataSet<T>(l * 2 / nr_class);
		int p = 0;
		for (int i = 0; i < nr_class; i++) {
			for (int j = i + 1; j < nr_class; j++) {
				final int si = start[i], sj = start[j];
				final int ci = count[i], cj = count[j];
				newx.clear();
				for (int k = 0, m = si; k < ci; ++k, ++m) {
					newx.add(x.get(perm[m]), +1);
				}
				for (int k = 0, m = sj; k < cj; ++k, ++m) {
					newx.add(x.get(perm[m]), -1);
				}

				if (probability) {
					double[] probAB = new double[2];
					svm_binary_svc_probability(newx, kf, weighted_C[i],
							weighted_C[j], probAB);
					probA[p] = probAB[0];
					probB[p] = probAB[1];
				}
				set_weights(weighted_C[i], weighted_C[j]);
				svm_train_one(newx, kf);
				f_alpha[p] = alpha;
				f_rho[p] = rho;
				for (int k = 0; k < ci; k++)
					if (!nonzero[si + k] && Math.abs(f_alpha[p][k]) > 0)
						nonzero[si + k] = true;
				for (int k = 0; k < cj; k++)
					if (!nonzero[sj + k] && Math.abs(f_alpha[p][ci + k]) > 0)
						nonzero[sj + k] = true;
				++p;
			}
		}

		// build output

		ClassificationModel<T> model = new ClassificationModel<T>();
		model.nr_class = nr_class;

		model.label = new int[nr_class];
		for (int i = 0; i < nr_class; i++)
			model.label[i] = label[i];

		model.rho = new double[pairs];
		for (int i = 0; i < pairs; i++)
			model.rho[i] = f_rho[i];

		if (probability) {
			ProbabilisticClassificationModel<T> pmodel = (ProbabilisticClassificationModel<T>) model;
			for (int i = 0; i < pairs; i++) {
				pmodel.probA[i] = probA[i];
				pmodel.probB[i] = probB[i];
			}
		}

		int nnz = 0;
		int[] nz_count = new int[nr_class];
		model.nSV = new int[nr_class];
		for (int i = 0; i < nr_class; i++) {
			int nSV = 0;
			for (int j = 0; j < count[i]; j++)
				if (nonzero[start[i] + j]) {
					++nSV;
					++nnz;
				}
			model.nSV[i] = nSV;
			nz_count[i] = nSV;
		}

		getLogger().info("Total nSV = " + nnz + "\n");

		model.l = nnz;
		model.SV = new ArrayList<T>(nnz);
		model.sv_indices = new int[nnz];
		p = 0;
		for (int i = 0; i < l; i++)
			if (nonzero[i]) {
				model.SV.add(x.get(perm[i]));
				model.sv_indices[p++] = perm[i] + 1;
			}

		int[] nz_start = new int[nr_class];
		nz_start[0] = 0;
		for (int i = 1; i < nr_class; i++)
			nz_start[i] = nz_start[i - 1] + nz_count[i - 1];

		model.sv_coef = new double[nr_class - 1][];
		for (int i = 0; i < nr_class - 1; i++)
			model.sv_coef[i] = new double[nnz];

		p = 0;
		for (int i = 0; i < nr_class; i++)
			for (int j = i + 1; j < nr_class; j++) {
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];

				int q = nz_start[i];
				for (int k = 0; k < ci; k++)
					if (nonzero[si + k])
						model.sv_coef[j - 1][q++] = f_alpha[p][k];
				q = nz_start[j];
				for (int k = 0; k < cj; k++)
					if (nonzero[sj + k])
						model.sv_coef[i][q++] = f_alpha[p][ci + k];
				++p;
			}
		return model;
	}

	// Stratified cross validation
	public void svm_cross_validation_classification(DataSet<T> x,
			KernelFunction<? super T> kf, double[] weighted_C, int nr_fold,
			double[] target) {
		final int l = x.size();
		int[] fold_start = new int[nr_fold + 1];
		int[] perm = new int[l];

		// stratified cv may not give leave-one-out rate
		// Each class to l folds -> some folds may have zero elements
		if (nr_fold < l) {
			stratifiedFolds(x, nr_fold, perm, fold_start);
		} else {
			perm = shuffledIndex(perm, l);
			// Split into folds
			for (int i = 0; i <= nr_fold; i++)
				fold_start[i] = i * l / nr_fold;
		}

		DoubleWeightedArrayDataSet<T> newx = new DoubleWeightedArrayDataSet<T>(
				l);
		for (int i = 0; i < nr_fold; i++) {
			int begin = fold_start[i];
			int end = fold_start[i + 1];

			newx.clear();
			for (int j = 0; j < begin; ++j) {
				newx.add(x.get(perm[j]), x.value(perm[j]));
			}
			for (int j = end; j < l; ++j) {
				newx.add(x.get(perm[j]), x.value(perm[j]));
			}
			ClassificationModel<T> submodel = svm_train_classification(newx,
					kf, weighted_C);
			if (submodel instanceof ProbabilisticClassificationModel) {
				ProbabilisticClassificationModel<T> pm = (ProbabilisticClassificationModel<T>) submodel;
				double[] prob_estimates = new double[submodel.nr_class];
				for (int j = begin; j < end; j++)
					target[perm[j]] = pm.predict_prob(x.get(perm[j]), kf,
							prob_estimates);
			} else {
				for (int j = begin; j < end; j++) {
					target[perm[j]] = submodel.predict(x.get(perm[j]), kf);
				}
			}
		}
	}

	// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
	private void sigmoid_train(double[] dec_values, DataSet<?> x,
			double[] probAB) {
		final int l = x.size();
		double A, B;
		double prior1 = 0, prior0 = 0;

		for (int i = 0; i < l; i++) {
			if (x.value(i) > 0) {
				++prior1;
			} else {
				++prior0;
			}
		}

		int max_iter = 100; // Maximal number of iterations
		double min_step = 1e-10; // Minimal step taken in line search
		double sigma = 1e-12; // For numerically strict PD of Hessian
		double eps = 1e-5;
		double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
		double loTarget = 1 / (prior0 + 2.0);
		double[] t = new double[l];
		double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
		double newA, newB, newf, d1, d2;

		// Initial Point and Initial Fun Value
		A = 0.0;
		B = Math.log((prior0 + 1.0) / (prior1 + 1.0));
		double fval = 0.0;

		for (int i = 0; i < l; i++) {
			t[i] = (x.value(i) > 0) ? hiTarget : loTarget;
			fApB = dec_values[i] * A + B;
			if (fApB >= 0) {
				fval += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
			} else {
				fval += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
			}
		}
		for (int iter = 0; iter < max_iter; iter++) {
			// Update Gradient and Hessian (use H' = H + sigma I)
			h11 = sigma; // numerically ensures strict PD
			h22 = sigma;
			h21 = 0.0;
			g1 = 0.0;
			g2 = 0.0;
			for (int i = 0; i < l; i++) {
				fApB = dec_values[i] * A + B;
				if (fApB >= 0) {
					p = Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
					q = 1.0 / (1.0 + Math.exp(-fApB));
				} else {
					p = 1.0 / (1.0 + Math.exp(fApB));
					q = Math.exp(fApB) / (1.0 + Math.exp(fApB));
				}
				d2 = p * q;
				h11 += dec_values[i] * dec_values[i] * d2;
				h22 += d2;
				h21 += dec_values[i] * d2;
				d1 = t[i] - p;
				g1 += dec_values[i] * d1;
				g2 += d1;
			}

			// Stopping Criteria
			if (Math.abs(g1) < eps && Math.abs(g2) < eps) {
				break;
			}

			// Finding Newton direction: -inv(H') * g
			det = h11 * h22 - h21 * h21;
			dA = -(h22 * g1 - h21 * g2) / det;
			dB = -(-h21 * g1 + h11 * g2) / det;
			gd = g1 * dA + g2 * dB;

			stepsize = 1; // Line Search
			while (stepsize >= min_step) {
				newA = A + stepsize * dA;
				newB = B + stepsize * dB;

				// New function value
				newf = 0.0;
				for (int i = 0; i < l; i++) {
					fApB = dec_values[i] * newA + newB;
					if (fApB >= 0) {
						newf += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
					} else {
						newf += (t[i] - 1) * fApB
								+ Math.log(1 + Math.exp(fApB));
					}
				}
				// Check sufficient decrease
				if (newf < fval + 0.0001 * stepsize * gd) {
					A = newA;
					B = newB;
					fval = newf;
					break;
				} else {
					stepsize = stepsize * .5;
				}
			}

			if (stepsize < min_step) {
				getLogger()
						.info("Line search fails in two-class probability estimates\n");
				break;
			}
			if (iter >= max_iter) {
				getLogger()
						.info("Reaching maximal iterations in two-class probability estimates\n");
			}
		}

		probAB[0] = A;
		probAB[1] = B;
	}

	// Cross-validation decision values for probability estimates
	private void svm_binary_svc_probability(DataSet<T> x,
			KernelFunction<? super T> kf, double Cp, double Cn, double[] probAB) {
		final int l = x.size();
		int nr_fold = 5;
		int[] perm = shuffledIndex(new int[l], l);
		double[] dec_values = new double[l];
		DataSet<T> newx = new ByteWeightedArrayDataSet<T>(l);
		for (int i = 0; i < nr_fold; i++) {
			int begin = i * l / nr_fold;
			int end = (i + 1) * l / nr_fold;

			newx.clear();
			for (int j = 0; j < begin; j++) {
				newx.add(x.get(perm[j]), x.value(perm[j]));
			}
			for (int j = end; j < l; j++) {
				newx.add(x.get(perm[j]), x.value(perm[j]));
			}
			int p_count = 0, n_count = 0;
			for (int j = 0; j < newx.size(); j++) {
				if (newx.value(j) > 0) {
					p_count++;
				} else {
					n_count++;
				}
			}

			if (p_count == 0 && n_count == 0) {
				for (int j = begin; j < end; j++) {
					dec_values[perm[j]] = 0;
				}
			} else if (p_count > 0 && n_count == 0) {
				for (int j = begin; j < end; j++) {
					dec_values[perm[j]] = 1;
				}
			} else if (p_count == 0 && n_count > 0) {
				for (int j = begin; j < end; j++) {
					dec_values[perm[j]] = -1;
				}
			} else {
				svm_parameter subparam = (svm_parameter) param.clone();
				subparam.probability = 0;
				subparam.C = 1.0;
				subparam.nr_weight = 2;
				subparam.weight_label = new int[2];
				subparam.weight = new double[2];
				subparam.weight_label[0] = +1;
				subparam.weight_label[1] = -1;
				subparam.weight[0] = Cp;
				subparam.weight[1] = Cn;
				ClassificationModel<T> submodel = (ClassificationModel<T>) svm_train_classification(
						newx, kf, new double[] { Cp, Cn });
				for (int j = begin; j < end; j++) {
					double[] dec_value = new double[1];
					submodel.predict(x.get(perm[j]), kf, dec_value);
					dec_values[perm[j]] = dec_value[0];
					// ensure +1 -1 order; reason not using CV subroutine
					dec_values[perm[j]] *= submodel.label[0];
				}
			}
		}
		sigmoid_train(dec_values, x, probAB);
	}
}
