package com.kno10.svm.libmodernsvm;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.format.libsvm.LibSVMDataReader;
import com.kno10.svm.libmodernsvm.format.libsvm.LibSVMModelWriter;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.LinearKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.offheap.OffHeapSparseVector;
import com.kno10.svm.libmodernsvm.kernelfunction.offheap.OffHeapSparseVectorBuilder;
import com.kno10.svm.libmodernsvm.model.ClassificationModel;
import com.kno10.svm.libmodernsvm.variants.AbstractSVC;
import com.kno10.svm.libmodernsvm.variants.CSVC;

/**
 * Simple application for debugging training.
 * 
 * TODO: parameters - develop into a fully compatible replacement for svm-train?
 * 
 * @author Erich Schubert
 */
public class SimpleTrainApp {

  public static void main(String[] args) {
    try {
      OffHeapSparseVectorBuilder b = new OffHeapSparseVectorBuilder();
      DataSet<OffHeapSparseVector> data = LibSVMDataReader.loadData(new FileInputStream(args[0]), b);
      double gamma = 1. / dimensionality(data); // Default: 1/numfeatures
      // KernelFunction<UnsafeSparseVector> kf = new
      // RadialBasisKernelFunction(gamma);
      KernelFunction<OffHeapSparseVector> kf = new LinearKernelFunction<OffHeapSparseVector>();
      System.err.println("Data set size: " + data.size());
      ClassificationModel<OffHeapSparseVector> m;
      AbstractSVC<OffHeapSparseVector> svm = new CSVC<OffHeapSparseVector>(1e-3, true, 2000);
      // AbstractSVC<UnsafeSparseVector> svm = new NuSVC<UnsafeSparseVector>(1,
      // true, 500, .5);
      m = svm.train(data, kf, null);
      System.err.println(m.l + " " + m.nr_class + " " + m.SV.size());

      LibSVMModelWriter.writeModel(new PrintWriter(new FileOutputStream(args[1])), m, svm, kf);
    }
    catch(IOException e) {
      e.printStackTrace();
    }
  }

  private static int dimensionality(DataSet<OffHeapSparseVector> data) {
    int max = -1;
    for(int i = 0; i < data.size(); ++i) {
      OffHeapSparseVector v = data.get(i);
      for(int j = 0, l = v.size(); j < l; ++j) {
        int idx = v.index(j);
        max = (idx > max) ? idx : max;
      }
    }
    return max + 1;
  }
}
