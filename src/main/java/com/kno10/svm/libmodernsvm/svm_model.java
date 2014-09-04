//
// svm_model
//
package com.kno10.svm.libmodernsvm;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

public class svm_model<T> implements java.io.Serializable {
	public int nr_class; // number of classes, = 2 in regression/one class svm
	public int l; // total #SV
	public T[] SV; // SVs (SV[l])
	public double[][] sv_coef; // coefficients for SVs in decision functions
								// (sv_coef[k-1][l])
	public double[] rho; // constants in decision functions (rho[k*(k-1)/2])
	public double[] probA; // pairwise probability information
	public double[] probB;
	public int[] sv_indices; // sv_indices[0,...,nSV-1] are values in
								// [1,...,num_traning_data] to indicate SVs in
								// the training set

	// for classification only

	public int[] label; // label of each class (label[k])
	public int[] nSV; // number of SVs for each class (nSV[k])
	static final String svm_type_table[] = { "c_svc", "nu_svc", "one_class",
			"epsilon_svr", "nu_svr", };

	static final String kernel_type_table[] = { "linear", "polynomial", "rbf",
			"sigmoid", "precomputed" };

	public int svm_get_nr_class() {
		return nr_class;
	}

	public void svm_get_labels(int[] label) {
		if (this.label != null)
			for (int i = 0; i < nr_class; i++)
				label[i] = this.label[i];
	}

	public void svm_get_sv_indices(int[] indices) {
		if (sv_indices != null)
			for (int i = 0; i < l; i++)
				indices[i] = this.sv_indices[i];
	}

	public int svm_get_nr_sv() {
		return this.l;
	}

	public void svm_save_model(svm_parameter param, String model_file_name)
			throws IOException {
		DataOutputStream fp = new DataOutputStream(new BufferedOutputStream(
				new FileOutputStream(model_file_name)));

		fp.writeBytes("svm_type " + svm_type_table[param.svm_type] + "\n");
		fp.writeBytes("kernel_type " + kernel_type_table[param.kernel_type]
				+ "\n");

		if (param.kernel_type == svm_parameter.POLY)
			fp.writeBytes("degree " + param.degree + "\n");

		if (param.kernel_type == svm_parameter.POLY
				|| param.kernel_type == svm_parameter.RBF
				|| param.kernel_type == svm_parameter.SIGMOID)
			fp.writeBytes("gamma " + param.gamma + "\n");

		if (param.kernel_type == svm_parameter.POLY
				|| param.kernel_type == svm_parameter.SIGMOID)
			fp.writeBytes("coef0 " + param.coef0 + "\n");

		fp.writeBytes("nr_class " + nr_class + "\n");
		fp.writeBytes("total_sv " + l + "\n");

		{
			fp.writeBytes("rho");
			for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
				fp.writeBytes(" " + rho[i]);
			fp.writeBytes("\n");
		}

		if (label != null) {
			fp.writeBytes("label");
			for (int i = 0; i < nr_class; i++)
				fp.writeBytes(" " + label[i]);
			fp.writeBytes("\n");
		}

		if (probA != null) // regression has probA only
		{
			fp.writeBytes("probA");
			for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
				fp.writeBytes(" " + probA[i]);
			fp.writeBytes("\n");
		}
		if (probB != null) {
			fp.writeBytes("probB");
			for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
				fp.writeBytes(" " + probB[i]);
			fp.writeBytes("\n");
		}

		if (nSV != null) {
			fp.writeBytes("nr_sv");
			for (int i = 0; i < nr_class; i++)
				fp.writeBytes(" " + nSV[i]);
			fp.writeBytes("\n");
		}

		fp.writeBytes("SV\n");
		for (int i = 0; i < l; i++) {
			for (int j = 0; j < nr_class - 1; j++)
				fp.writeBytes(sv_coef[j][i] + " ");

			// FIXME: cast specific to libSVM data model
			svm_node[] p = (svm_node[]) SV[i];
			if (param.kernel_type == svm_parameter.PRECOMPUTED)
				fp.writeBytes("0:" + (int) (p[0].value));
			else
				for (int j = 0; j < p.length; j++)
					fp.writeBytes(p[j].index + ":" + p[j].value + " ");
			fp.writeBytes("\n");
		}

		fp.close();
	}

	private static boolean read_model_header(BufferedReader fp,
			svm_parameter param, svm_model<svm_node[]> model) {
		try {
			while (true) {
				String cmd = fp.readLine();
				String arg = cmd.substring(cmd.indexOf(' ') + 1);

				if (cmd.startsWith("svm_type")) {
					int i;
					for (i = 0; i < svm_type_table.length; i++) {
						if (arg.indexOf(svm_type_table[i]) != -1) {
							param.svm_type = i;
							break;
						}
					}
					if (i == svm_type_table.length) {
						System.err.print("unknown svm type.\n");
						return false;
					}
				} else if (cmd.startsWith("kernel_type")) {
					int i;
					for (i = 0; i < kernel_type_table.length; i++) {
						if (arg.indexOf(kernel_type_table[i]) != -1) {
							param.kernel_type = i;
							break;
						}
					}
					if (i == kernel_type_table.length) {
						System.err.print("unknown kernel function.\n");
						return false;
					}
				} else if (cmd.startsWith("degree"))
					param.degree = Integer.parseInt(arg);
				else if (cmd.startsWith("gamma"))
					param.gamma = Double.parseDouble(arg);
				else if (cmd.startsWith("coef0"))
					param.coef0 = Double.parseDouble(arg);
				else if (cmd.startsWith("nr_class"))
					model.nr_class = Integer.parseInt(arg);
				else if (cmd.startsWith("total_sv"))
					model.l = Integer.parseInt(arg);
				else if (cmd.startsWith("rho")) {
					int n = model.nr_class * (model.nr_class - 1) / 2;
					model.rho = new double[n];
					StringTokenizer st = new StringTokenizer(arg);
					for (int i = 0; i < n; i++)
						model.rho[i] = Double.parseDouble(st.nextToken());
				} else if (cmd.startsWith("label")) {
					int n = model.nr_class;
					model.label = new int[n];
					StringTokenizer st = new StringTokenizer(arg);
					for (int i = 0; i < n; i++)
						model.label[i] = Integer.parseInt(st.nextToken());
				} else if (cmd.startsWith("probA")) {
					int n = model.nr_class * (model.nr_class - 1) / 2;
					model.probA = new double[n];
					StringTokenizer st = new StringTokenizer(arg);
					for (int i = 0; i < n; i++)
						model.probA[i] = Double.parseDouble(st.nextToken());
				} else if (cmd.startsWith("probB")) {
					int n = model.nr_class * (model.nr_class - 1) / 2;
					model.probB = new double[n];
					StringTokenizer st = new StringTokenizer(arg);
					for (int i = 0; i < n; i++)
						model.probB[i] = Double.parseDouble(st.nextToken());
				} else if (cmd.startsWith("nr_sv")) {
					int n = model.nr_class;
					model.nSV = new int[n];
					StringTokenizer st = new StringTokenizer(arg);
					for (int i = 0; i < n; i++)
						model.nSV[i] = Integer.parseInt(st.nextToken());
				} else if (cmd.startsWith("SV")) {
					break;
				} else {
					System.err.print("unknown text in model file: [" + cmd
							+ "]\n");
					return false;
				}
			}
		} catch (Exception e) {
			return false;
		}
		return true;
	}

	public static svm_model<svm_node[]> svm_load_model(String model_file_name)
			throws IOException {
		return svm_load_model(new BufferedReader(
				new FileReader(model_file_name)));
	}

	public static svm_model<svm_node[]> svm_load_model(BufferedReader fp)
			throws IOException {
		// read parameters

		svm_parameter param = new svm_parameter();
		svm_model<svm_node[]> model = new svm_model<svm_node[]>();
		model.rho = null;
		model.probA = null;
		model.probB = null;
		model.label = null;
		model.nSV = null;

		if (read_model_header(fp, param, model) == false) {
			System.err.print("ERROR: failed to read model\n");
			return null;
		}

		// read sv_coef and SV

		int m = model.nr_class - 1;
		int l = model.l;
		model.sv_coef = new double[m][l];
		model.SV = new svm_node[l][];

		for (int i = 0; i < l; i++) {
			String line = fp.readLine();
			StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

			for (int k = 0; k < m; k++)
				model.sv_coef[k][i] = Double.parseDouble(st.nextToken());
			int n = st.countTokens() / 2;
			model.SV[i] = new svm_node[n];
			for (int j = 0; j < n; j++) {
				model.SV[i][j] = new svm_node();
				model.SV[i][j].index = Integer.parseInt(st.nextToken());
				model.SV[i][j].value = Double.parseDouble(st.nextToken());
			}
		}

		fp.close();
		return model;
	}

	// nSV[0] + nSV[1] + ... + nSV[k-1] = l

	public int svm_check_probability_model(svm_parameter param) {
		if (((param.svm_type == svm_parameter.C_SVC || param.svm_type == svm_parameter.NU_SVC)
				&& probA != null && probB != null)
				|| ((param.svm_type == svm_parameter.EPSILON_SVR || param.svm_type == svm_parameter.NU_SVR) && probA != null))
			return 1;
		else
			return 0;
	}
};
