package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.ArrayUtil;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.variants.CSVC;
import com.kno10.svm.libmodernsvm.variants.NuSVC;

/**
 * Q matrix used by {@link CSVC} and {@link NuSVC} classification.
 *
 * @param <T>
 */
public class SVC_Q extends Kernel {
  public <T> SVC_Q(final DataSet<T> x, final KernelFunction<? super T> kf, double cache_size, final byte[] y) {
    // TODO: clone y removal okay?
    super(new Cache<T>(x.size(), (long) (cache_size * (1 << 20))) {
      @Override
      public double similarity(int i, int j) {
        return y[i] * y[j] * kf.similarity(x.get(i), x.get(j));
      }

      @Override
      void swap_index(int i, int j) {
        if(i == j) {
          return;
        }
        super.swap_index(i, j);
        x.swap(i, j);
        ArrayUtil.swap(y, i, j);
      }
    }, x.size());
  }
}
