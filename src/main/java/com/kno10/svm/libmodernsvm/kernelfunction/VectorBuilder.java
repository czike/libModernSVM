package com.kno10.svm.libmodernsvm.kernelfunction;

/**
 * Build a vector from data.
 * 
 * @author Erich Schubert
 *
 * @param <V> Vector type
 */
public interface VectorBuilder<V extends Vector<V>> {
  /**
   * Reset the builder buffer.
   */
  void clear();

  /**
   * Add a new value.
   * 
   * @param idx Index, starting at 0
   * @param val Value
   */
  void add(int idx, double val);

  /**
   * Build the vector object.
   * 
   * @return New vector
   */
  V build();
}
