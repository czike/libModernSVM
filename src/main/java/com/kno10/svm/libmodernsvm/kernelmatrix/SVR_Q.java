package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.ArrayUtil;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.variants.EpsilonSVR;
import com.kno10.svm.libmodernsvm.variants.NuSVR;

/**
 * Q matrix used for regression by {@link EpsilonSVR} and {@link NuSVR}.
 * 
 * This uses two "copies" of the data, one for upper bounding and one for lower
 * bounding of the data, yielding a virtual size of 2*l.
 *
 * @param <T>
 */
public class SVR_Q<T> extends Kernel<T> {
  private final int l;

  private final byte[] sign;

  private final int[] index;

  private final double[] QD;

  private final float[] tmp;

  public SVR_Q(DataSet<T> x, KernelFunction<? super T> kf, double cache_size) {
    super(x, kf, cache_size);
    this.l = x.size();
    QD = new double[l << 1];
    sign = new byte[l << 1];
    index = new int[l << 1];
    for(int k = 0, k2 = l; k < l; k++, k2++) {
      sign[k] = 1;
      sign[k2] = -1;
      index[k] = k;
      index[k2] = k;
      QD[k] = similarity(k, k);
      QD[k2] = QD[k];
    }
    this.tmp = new float[l];
  }

  @Override
  public void swap_index(int i, int j) {
    // Note: the cache was not reordered!
    ArrayUtil.swap(sign, i, j);
    ArrayUtil.swap(index, i, j);
    ArrayUtil.swap(QD, i, j);
  }

  @Override
  public void get_Q(int i, int len, float[] out) {
    final int real_i = index[i];
    // From cache, not reordered; always get all l values!
    super.get_Q(real_i, l, tmp);

    // reorder and copy to output
    final byte si = sign[i];
    for(int j = 0; j < len; j++) {
      out[j] = (float) si * sign[j] * tmp[index[j]];
    }
  }

  @Override
  public double[] get_QD() {
    return QD;
  }
}