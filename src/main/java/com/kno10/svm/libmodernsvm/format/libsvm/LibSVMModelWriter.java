package com.kno10.svm.libmodernsvm.format.libsvm;

import java.io.IOException;
import java.io.PrintWriter;

import javolution.text.TextBuilder;
import javolution.text.TypeFormat;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.unsafe.LinearKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.unsafe.PolynomialKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.unsafe.RadialBasisKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.unsafe.UnsafeSparseVector;
import com.kno10.svm.libmodernsvm.model.ClassificationModel;
import com.kno10.svm.libmodernsvm.variants.AbstractSVC;
import com.kno10.svm.libmodernsvm.variants.CSVC;
import com.kno10.svm.libmodernsvm.variants.NuSVC;

public class LibSVMModelWriter {
  public static void writeModel(PrintWriter p, ClassificationModel<UnsafeSparseVector> m, AbstractSVC<UnsafeSparseVector> svm, KernelFunction<UnsafeSparseVector> kf) throws IOException {
    TextBuilder tb = new TextBuilder();
    // FIXME: make extensible?
    if(svm instanceof CSVC) {
      p.println("svm_type c_svc");
    }
    else if(svm instanceof NuSVC) {
      p.println("svm_type nu_svc");
    }
    else {
      System.err.println("Unknown SVM type: " + svm.toString());
    }
    // FIXME: make extensible
    if(kf instanceof LinearKernelFunction) {
      p.println("kernel_type linear");
    }
    else if(kf instanceof PolynomialKernelFunction) {
      p.println("kernel_type polynomial");
      p.println("degree " + ((PolynomialKernelFunction) kf).degree());
      p.println("gamma" + ((PolynomialKernelFunction) kf).gamma());
      p.println("coeff0 " + ((PolynomialKernelFunction) kf).coeff0());
    }
    else if(kf instanceof RadialBasisKernelFunction) {
      p.println("kernel_type rbf");
      p.println("gamma " + ((RadialBasisKernelFunction) kf).gamma());
    }
    else {
      System.err.println("Unknown kernel function: " + kf.toString());
    }
    p.println("nr_class " + m.nr_class);
    p.println("total_sv " + m.SV.size());
    tb.clear();
    tb.append("rho");
    for(int i = 0; i < m.rho.length; i++) {
      tb.append(' ');
      tb.append((float) m.rho[i]);
    }
    p.println(tb);
    tb.clear();
    tb.append("label");
    for(int i = 0; i < m.label.length; i++) {
      tb.append(' ');
      tb.append(m.label[i]);
    }
    p.println(tb);
    tb.clear();
    tb.append("nr_sv");
    for(int i = 0; i < m.nSV.length; i++) {
      tb.append(' ');
      tb.append(m.nSV[i]);
    }
    p.println(tb);
    p.println("SV");
    for(int j = 0, n = m.SV.size(); j < n; ++j) {
      tb.clear();
      // Print all but last one:
      for(int i = 0; i < m.nr_class - 1; i++) {
        if(i > 0) {
          tb.append(' ');
        }
        tb.append(m.sv_coef[i][j], 16, false, false);
      }
      // Print vector
      UnsafeSparseVector sv = m.SV.get(j);
      for(int i = 0, l = sv.size(); i < l; ++i) {
        tb.append(' ');
        TypeFormat.format(sv.index(i) + 1, tb);
        tb.append(':');
        append(sv.value(i), tb);
      }
      tb.append(' '); // Compatibility with libsvm output
      p.println(tb);
    }
    p.close();
  }

  private static void append(double v, TextBuilder tb) {
    if(v == 1.) {
      tb.append('1');
    }
    else if(v == -1.) {
      tb.append("-1");
    }
    else {
      tb.append((float) v, 6, false, false);
    }
  }
}
