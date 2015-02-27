package com.kno10.svm.libmodernsvm.kernelmatrix;

import com.kno10.svm.libmodernsvm.ArrayUtil;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.variants.QMatrix;

public class Kernel implements QMatrix {
  protected static final class KernelCache<T> extends Cache<T> {
    private final DataSet<T> x;

    private final KernelFunction<? super T> kf;

    protected KernelCache(DataSet<T> x, KernelFunction<? super T> kf, double cache_size) {
      super(x.size(), cache_size);
      this.x = x;
      this.kf = kf;
    }

    @Override
    public double similarity(int i, int j) {
      return kf.similarity(x.get(i), x.get(j));
    }

    @Override
    void swap_index(int i, int j) {
      if(i == j) {
        return;
      }
      super.swap_index(i, j);
      x.swap(i, j);
    }
  }

  protected final Cache<?> cache;

  // Diagonal values <x,x>
  protected final double[] QD;

  public Kernel(Cache<?> cache, int l) {
    super();
    this.cache = cache;
    QD = initializeQD(l);
  }

  public <T> Kernel(final DataSet<T> x, final KernelFunction<? super T> kf, double cache_size) {
    this(new KernelCache<T>(x, kf, cache_size), x.size());
  }

  protected double[] initializeQD(final int l) {
    double[] QD = new double[l];
    for(int i = 0; i < l; i++) {
      QD[i] = cache.similarity(i, i);
    }
    return QD;
  }

  public void swap_index(int i, int j) {
    // Swap in cache, too:
    cache.swap_index(i, j);
    ArrayUtil.swap(QD, i, j);
  }

  @Override
  public final double[] get_QD() {
    return QD;
  }

  public void get_Q(int i, int len, float[] out) {
    float[] data = cache.get_data(i, len);
    if(out != null)
      System.arraycopy(data, 0, out, 0, len);
  }

  @Override
  public double quadDistance(int i, int j, byte b) {
    float[] Q_i = cache.get_data(i, j + 1);
    return QD[i] + QD[j] - 2.0 * b * Q_i[j];
  }
}