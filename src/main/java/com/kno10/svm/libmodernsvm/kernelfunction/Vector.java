package com.kno10.svm.libmodernsvm.kernelfunction;

/**
 * Interface of sparse vectors.
 * 
 * @author Erich Schubert
 *
 * @param <SELF>
 */
public interface Vector<SELF> {
  /**
   * Number of valid entries in this sparse vector.
   * 
   * @return Size
   */
  public int size();

  /**
   * Index at position i.
   * 
   * <i>Warning:</i> does not check bounds. This can crash your VM.
   * 
   * @param i Position
   * @return Index
   */
  public int index(int i);

  /**
   * Value at position i
   *
   * @param i Position
   * @return Value
   */
  public double value(int i);

  /**
   * Dot product of two vectors. Low level.
   * 
   * @param y Second vector
   * @return Dot product
   */
  public double dot(SELF y);

  /**
   * Squared Euclidean distance of two vectors.
   * 
   * @param y Second vector
   * @return Squared Euclidean distance
   */
  double squareEuclidean(SELF y);
}
