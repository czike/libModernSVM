//
// svm_model
//
package com.kno10.svm.libmodernsvm.model;

import java.util.ArrayList;

public class Model<T> {
	public int nr_class; // number of classes, = 2 in regression/one class svm
	public int l; // total #SV
	public ArrayList<T> SV; // SVs (SV[l])
	public double[][] sv_coef; // coefficients for SVs in decision functions
								// (sv_coef[k-1][l])
	public double[] rho; // constants in decision functions (rho[k*(k-1)/2])
	public int[] sv_indices; // sv_indices[0,...,nSV-1] are values in
								// [1,...,num_traning_data] to indicate SVs in
								// the training set
};
