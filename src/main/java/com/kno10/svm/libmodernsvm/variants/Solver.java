package com.kno10.svm.libmodernsvm.variants;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.ArrayUtil;

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
public class Solver {
  private static final Logger LOG = Logger.getLogger(Solver.class.getName());

  int active_size;

  byte[] y;

  double[] G; // gradient of objective function

  static final byte LOWER_BOUND = 0;

  static final byte UPPER_BOUND = 1;

  static final byte FREE = 2;

  byte[] alpha_status; // LOWER_BOUND, UPPER_BOUND, FREE

  double[] alpha;

  QMatrix Q;

  double[] QD;

  double eps;

  double Cp, Cn;

  double[] p;

  int[] active_set;

  double[] G_bar; // gradient, if we treat free variables as 0

  int l;

  boolean unshrink; // XXX

  double get_C(int i) {
    return (y[i] > 0) ? Cp : Cn;
  }

  void update_alpha_status(int i) {
    if(alpha[i] >= get_C(i)) {
      alpha_status[i] = UPPER_BOUND;
    }
    else if(alpha[i] <= 0) {
      alpha_status[i] = LOWER_BOUND;
    }
    else {
      alpha_status[i] = FREE;
    }
  }

  boolean is_upper_bound(int i) {
    return alpha_status[i] == UPPER_BOUND;
  }

  boolean is_lower_bound(int i) {
    return alpha_status[i] == LOWER_BOUND;
  }

  boolean is_free(int i) {
    return alpha_status[i] == FREE;
  }

  // java: information about solution except alpha,
  // because we cannot return multiple values otherwise...
  static class SolutionInfo {
    public SolutionInfo(int l) {
      alpha = new double[l];
    }

    double obj, rho;

    double upper_bound_p, upper_bound_n;

    double r; // for Solver_NU

    double[] alpha;
  }

  void swap_index(int i, int j) {
    // This will also swap QD:
    Q.swap_index(i, j);
    ArrayUtil.swap(y, i, j);
    ArrayUtil.swap(G, i, j);
    ArrayUtil.swap(alpha_status, i, j);
    ArrayUtil.swap(alpha, i, j);
    ArrayUtil.swap(p, i, j);
    ArrayUtil.swap(active_set, i, j);
    ArrayUtil.swap(G_bar, i, j);
  }

  void reconstruct_gradient() {
    // reconstruct inactive elements of G from G_bar and free variables
    if(active_size == l) {
      return;
    }

    for(int j = active_size; j < l; j++) {
      G[j] = G_bar[j] + p[j];
    }

    int nr_free = 0;
    for(int j = 0; j < active_size; j++) {
      if(is_free(j)) {
        nr_free++;
      }
    }

    if(2 * nr_free < active_size) {
      LOG.info("WARNING: using -h 0 may be faster");
    }

    if(nr_free * l > 2 * active_size * (l - active_size)) {
      float[] Q_i = new float[active_size];
      for(int i = active_size; i < l; i++) {
        Q.get_Q(i, active_size, Q_i);
        for(int j = 0; j < active_size; j++) {
          if(is_free(j)) {
            G[i] += alpha[j] * Q_i[j];
          }
        }
      }
    }
    else {
      float[] Q_i = new float[l];
      for(int i = 0; i < active_size; i++) {
        if(is_free(i)) {
          Q.get_Q(i, l, Q_i);
          double alpha_i = alpha[i];
          for(int j = active_size; j < l; j++) {
            G[j] += alpha_i * Q_i[j];
          }
        }
      }
    }
  }

  SolutionInfo solve(int l, QMatrix Q, double[] p_, byte[] y_, double[] alpha_, double Cp, double Cn, double eps, boolean shrinking) {
    SolutionInfo si = new SolutionInfo(l);
    solve(si, l, Q, p_, y_, alpha_, Cp, Cn, eps, shrinking);
    return si;
  }

  void solve(SolutionInfo si, int l, QMatrix Q, double[] p_, byte[] y_, double[] alpha_, double Cp, double Cn, double eps, boolean shrinking) {
    this.l = l;
    this.Q = Q;
    this.QD = Q.get_QD();
    this.p = p_.clone();
    this.y = y_.clone();
    this.alpha = alpha_.clone();
    this.Cp = Cp;
    this.Cn = Cn;
    this.eps = eps;
    this.unshrink = false;

    // initialize alpha_status
    {
      alpha_status = new byte[l];
      for(int i = 0; i < l; i++) {
        update_alpha_status(i);
      }
    }

    // initialize active set (for shrinking)
    {
      active_set = new int[l];
      for(int i = 0; i < l; i++) {
        active_set[i] = i;
      }
      active_size = l;
    }

    // initialize gradient
    initializeGradient();

    // optimization step

    int max_iter = Math.max(10000000, l > Integer.MAX_VALUE / 100 ? Integer.MAX_VALUE : 100 * l);
    int counter = Math.min(l, 1000) + 1;
    int[] working_set = new int[2];

    float[] Q_i = new float[l], Q_j = new float[l];
    int iter;
    for(iter = 0; iter < max_iter; ++iter) {
      // show progress and do shrinking

      if(--counter == 0) {
        counter = Math.min(l, 1000);
        if(shrinking) {
          do_shrinking();
        }
      }

      if(select_working_set(working_set) != 0) {
        // reconstruct the whole gradient
        reconstruct_gradient();
        // reset active set size and check
        active_size = l;
        if(select_working_set(working_set) != 0) {
          break;
        }
        counter = 1; // do shrinking next iteration
      }

      int i = working_set[0], j = working_set[1];

      // update alpha[i] and alpha[j], handle bounds carefully

      Q.get_Q(i, active_size, Q_i);
      Q.get_Q(j, active_size, Q_j);

      double C_i = get_C(i), C_j = get_C(j);

      double old_alpha_i = alpha[i], old_alpha_j = alpha[j];

      if(y[i] != y[j]) {
        double quad_coef = QD[i] + QD[j] + 2 * Q_i[j];
        double delta = (-G[i] - G[j]) / nonzero(quad_coef);
        double diff = alpha[i] - alpha[j];
        alpha[i] += delta;
        alpha[j] += delta;

        if(diff > 0) {
          if(alpha[j] < 0) {
            alpha[j] = 0;
            alpha[i] = diff;
          }
        }
        else {
          if(alpha[i] < 0) {
            alpha[i] = 0;
            alpha[j] = -diff;
          }
        }
        if(diff > C_i - C_j) {
          if(alpha[i] > C_i) {
            alpha[i] = C_i;
            alpha[j] = C_i - diff;
          }
        }
        else {
          if(alpha[j] > C_j) {
            alpha[j] = C_j;
            alpha[i] = C_j + diff;
          }
        }
      }
      else {
        double quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
        double delta = (G[i] - G[j]) / nonzero(quad_coef);
        double sum = alpha[i] + alpha[j];
        alpha[i] -= delta;
        alpha[j] += delta;

        if(sum > C_i) {
          if(alpha[i] > C_i) {
            alpha[i] = C_i;
            alpha[j] = sum - C_i;
          }
        }
        else {
          if(alpha[j] < 0) {
            alpha[j] = 0;
            alpha[i] = sum;
          }
        }
        if(sum > C_j) {
          if(alpha[j] > C_j) {
            alpha[j] = C_j;
            alpha[i] = sum - C_j;
          }
        }
        else {
          if(alpha[i] < 0) {
            alpha[i] = 0;
            alpha[j] = sum;
          }
        }
      }

      // update G
      double delta_alpha_i = alpha[i] - old_alpha_i;
      double delta_alpha_j = alpha[j] - old_alpha_j;
      for(int k = 0; k < active_size; k++) {
        G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
      }

      // update alpha_status and G_bar
      boolean ui = is_upper_bound(i), uj = is_upper_bound(j);
      update_alpha_status(i);
      update_alpha_status(j);
      if(ui != is_upper_bound(i)) {
        Q.get_Q(i, l, Q_i);
        update_G_bar(ui ? -C_i : C_i, Q_i, l);
      }

      if(uj != is_upper_bound(j)) {
        Q.get_Q(j, l, Q_j);
        update_G_bar(uj ? -C_j : C_j, Q_j, l);
      }
      if(iter >= max_iter) {
        if(active_size < l) {
          // reconstruct the whole gradient to calculate objective
          // value
          reconstruct_gradient();
          active_size = l;
        }
        System.err.print("\nWARNING: reaching max number of iterations\n");
      }
    }
    if(LOG.isLoggable(Level.INFO)) {
      LOG.log(Level.INFO, "optimization finished, #iter = {0}", iter);
    }

    // calculate rho
    si.rho = calculate_rho();

    // calculate objective value
    si.obj = calculate_obj();

    // put back the solution
    for(int i = 0; i < l; i++) {
      si.alpha[active_set[i]] = alpha[i];
    }

    si.upper_bound_p = Cp;
    si.upper_bound_n = Cn;
  }

  private void update_G_bar(double C_i, float[] Q_i, int l) {
    for(int k = 0; k < l; k++) {
      G_bar[k] += C_i * Q_i[k];
    }
  }

  public void initializeGradient() {
    G = new double[l];
    G_bar = new double[l];
    for(int i = 0; i < l; i++) {
      G[i] = p[i];
      G_bar[i] = 0;
    }
    float[] Q_i = new float[l];
    for(int i = 0; i < l; i++) {
      if(!is_lower_bound(i)) {
        Q.get_Q(i, l, Q_i);
        double alpha_i = alpha[i];
        for(int j = 0; j < l; j++) {
          G[j] += alpha_i * Q_i[j];
        }
        if(is_upper_bound(i)) {
          for(int j = 0; j < l; j++) {
            G_bar[j] += get_C(i) * Q_i[j];
          }
        }
      }
    }
  }

  protected double calculate_obj() {
    double v = 0.;
    for(int i = 0; i < l; i++) {
      v += alpha[i] * (G[i] + p[i]);
    }
    return v * .5;
  }

  protected static double nonzero(double d) {
    return d > 0 ? d : 1e-12;
  }

  // return 1 if already optimal, return 0 otherwise
  int select_working_set(int[] working_set) {
    // return i,j such that
    // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    // j: mimimizes the decrease of obj value
    // (if quadratic coefficeint <= 0, replace it with tau)
    // -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

    double Gmax = Double.NEGATIVE_INFINITY, Gmax2 = Double.NEGATIVE_INFINITY;
    int Gmax_idx = -1, Gmin_idx = -1;
    double obj_diff_min = Double.POSITIVE_INFINITY;

    for(int t = 0; t < active_size; t++) {
      if(y[t] == +1) {
        if(!is_upper_bound(t)) {
          if(-G[t] >= Gmax) {
            Gmax = -G[t];
            Gmax_idx = t;
          }
        }
      }
      else {
        if(!is_lower_bound(t)) {
          if(G[t] >= Gmax) {
            Gmax = G[t];
            Gmax_idx = t;
          }
        }
      }
    }

    int i = Gmax_idx;
    float[] Q_i = null;
    if(i != -1) { // null Q_i not accessed: Gmax=-INF if i=-1
      Q_i = new float[active_size];
      Q.get_Q(i, active_size, Q_i);
    }

    for(int j = 0; j < active_size; j++) {
      if(y[j] == +1) {
        if(!is_lower_bound(j)) {
          double grad_diff = Gmax + G[j];
          if(G[j] >= Gmax2) {
            Gmax2 = G[j];
          }
          if(grad_diff > 0) {
            double quad_coef = QD[i] + QD[j] - 2.0 * y[i] * Q_i[j];
            double obj_diff = -(grad_diff * grad_diff) / nonzero(quad_coef);

            if(obj_diff <= obj_diff_min) {
              Gmin_idx = j;
              obj_diff_min = obj_diff;
            }
          }
        }
      }
      else {
        if(!is_upper_bound(j)) {
          double grad_diff = Gmax - G[j];
          if(-G[j] >= Gmax2) {
            Gmax2 = -G[j];
          }
          if(grad_diff > 0) {
            double quad_coef = QD[i] + QD[j] + 2.0 * y[i] * Q_i[j];
            double obj_diff = -(grad_diff * grad_diff) / nonzero(quad_coef);

            if(obj_diff <= obj_diff_min) {
              Gmin_idx = j;
              obj_diff_min = obj_diff;
            }
          }
        }
      }
    }

    if(Gmax + Gmax2 < eps) {
      return 1;
    }

    working_set[0] = Gmax_idx;
    working_set[1] = Gmin_idx;
    return 0;
  }

  private boolean be_shrunk(int i, double Gmax1, double Gmax2) {
    if(is_upper_bound(i)) {
      return (y[i] == +1) ? (-G[i] > Gmax1) : (-G[i] > Gmax2);
    }
    if(is_lower_bound(i)) {
      return (y[i] == +1) ? (G[i] > Gmax2) : (G[i] > Gmax1);
    }
    return false;
  }

  void do_shrinking() {
    double Gmax1 = Double.NEGATIVE_INFINITY; // max { -y_i * grad(f)_i | i
    // in I_up(\alpha) }
    double Gmax2 = Double.NEGATIVE_INFINITY; // max { y_i * grad(f)_i | i in
    // I_low(\alpha) }

    // find maximal violating pair first
    for(int i = 0; i < active_size; i++) {
      if(y[i] == +1) {
        if(!is_upper_bound(i)) {
          if(-G[i] >= Gmax1) {
            Gmax1 = -G[i];
          }
        }
        if(!is_lower_bound(i)) {
          if(G[i] >= Gmax2) {
            Gmax2 = G[i];
          }
        }
      }
      else {
        if(!is_upper_bound(i)) {
          if(-G[i] >= Gmax2) {
            Gmax2 = -G[i];
          }
        }
        if(!is_lower_bound(i)) {
          if(G[i] >= Gmax1) {
            Gmax1 = G[i];
          }
        }
      }
    }

    if(unshrink == false && Gmax1 + Gmax2 <= eps * 10) {
      unshrink = true;
      reconstruct_gradient();
      active_size = l;
    }

    for(int i = 0; i < active_size; i++) {
      if(be_shrunk(i, Gmax1, Gmax2)) {
        for(active_size--; active_size > i; active_size--) {
          if(!be_shrunk(active_size, Gmax1, Gmax2)) {
            swap_index(i, active_size);
            break;
          }
        }
      }
    }
  }

  double calculate_rho() {
    int nr_free = 0;
    double ub = Double.POSITIVE_INFINITY, lb = Double.NEGATIVE_INFINITY, sum_free = 0;
    for(int i = 0; i < active_size; i++) {
      final double yG = y[i] * G[i];
      if(is_lower_bound(i)) {
        if(y[i] > 0) {
          ub = ub < yG ? ub : yG;
        }
        else {
          lb = lb > yG ? lb : yG;
        }
      }
      else if(is_upper_bound(i)) {
        if(y[i] < 0) {
          ub = ub < yG ? ub : yG;
        }
        else {
          lb = lb > yG ? lb : yG;
        }
      }
      else {
        ++nr_free;
        sum_free += yG;
      }
    }

    return (nr_free > 0) ? sum_free / nr_free : (ub + lb) * .5;
  }
}
