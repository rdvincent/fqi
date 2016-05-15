#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <complex>

using namespace std;

#include "domain.h"
#include "random.h"

class TassModel: public Domain {
public:
  int nstate;
private:
  int N;
  double *y;
  double omega;
  double K;
  double D;
  double I;
  int nn;
  int C;
  double hsyncp;
  static const double dt = 0.05;
  double *tlast;
  bool *fired;
  complex<double> i;
  vector<complex<double> > Z1;
  double R1;
  double dR1;
  int timestep;                 /* time step from initialization */
  complex<double> Zn[4];

private:
  void checkfired() {
    for (int j = 0; j < N; j++) {
      if (cos(y[j]) > 0.99) {
        fired[j] = true;
      }
    }
  }

  int countfired() {
    int nfired = 0;
    for (int j = 0; j < N; j++) {
      if (fired[j]) {
        nfired++;
        fired[j] = false;
      }
    }
    return nfired;
  }

  void checkrange() {
    for (int j = 0; j < N; j++) {
      if (y[j] < 0) {
        y[j] = y[j] + 2.0*M_PI;
      }
      else if (y[j] > 2.0*M_PI) {
        y[j] = y[j] - 2.0*M_PI;
      }
    }
  }

public:
  TassModel(int noscillators=100, double F = 1.0) {
    numActions = 9;
    numSteps = 1000;
    numDimensions = 7;

    omega = F*2.0*M_PI;
    K = 2.0;
    I = 30.0;
    D = 0.4;
    N = noscillators;
    hsyncp=100;
    y = new double[N];
    fired = new bool[N];
    nstate = 10;                /* R1, dR1, 4 delay, 4 R1_n */

    for (int j = 0; j < N; j++) {
      y[j] = rnd1() * 2.0 * M_PI;
    }

    checkrange();
    /* for complex math */
    i = sqrt(complex<double>(-1));
    tlast = new double[4];
    for (int j = 0; j < 4; j++) {
      tlast[j] = 0.0;
    }
    C = round(1.0/0.01);
    Z1.resize(C, 0.0);
    nn = 0;
    timestep = 0;
  }

  double getdt() {
    return dt;
  }

  vector<double> initialState()
  {
    vector<double> s;
    return s;
  }

  void setFeatures(int n) {
    switch (n) {
    case 2:                     /* just R1, dR1 */
    case 6:                     /* just R1, dR1, 4x R1_n */
    case 9:                     /* leave out dR1 */
    case 10:                    /* full vector */
      nstate = n;
      break;
    default:
      cerr << "Illegal number of features " << n << endl;
      break;
    }
  }

  void getValues(double *v) {
    *v++ = R1;

    if (nstate != 9) {
      *v++ = dR1;
    }

    if (nstate >= 9) {
      for (int i = 0; i < 4; i++) {
        *v++ = timestep * dt - tlast[i];
      }
    }

    if (nstate >= 6) {
      for (int i = 0; i < 4; i++) {
        *v++ = abs(Zn[i]);
      }
    }
  }

  double getReward(vector<double> s, int a) const {
    double r = 0;
    /* Penalize stimulation.
     */
    for (int k = 0; k < 4; k++) {
      if ((a & (1 << k)) != 0) {
        r -= N / 4.0;
      }
    }
    r /= N;
    if (nn >= N/3) {
      r += -hsyncp;             /* penalize simultaneous firing */
    }
    else {
      r += (double)nn/N;        /* small reward for firing! */
    }
    return r;
  }

  static complex<double> complex_from_polar(double r, double theta) {
    complex<double> y(r * cos(theta), r * sin(theta));
    return y;
  }

  static complex<double> * dft(vector<complex<double> > x) {
    int N = x.size();
    complex<double> *y = new complex<double>[N];
    complex<double> *r = new complex<double>[N];
    int k, n;
    for (k = 0; k < N; k++) {
      r[k] = complex_from_polar(1.0, -2.0*M_PI*(double)k/(double)N);
    }
    for (k = 0; k < N; k++) {
      y[k] = 0;
      for (n = 0; n < N; n++) {
        y[k] += x[n] * r[(n * k) % N];
      }
    }
    delete r;
    return y;
  }

  OneStepResult performAction(vector<double> s, int a) {
    int X[N];
    const double real_dt = 0.01;

    for (int k = 0; k < N; k++) {
      X[k] = 0;
    }
    for (int k = 0; k < 4; k++) {
      if ((a & (1 << k)) != 0) { 
        int first = (N * k) / 4;
        int last = (N * (k + 1)) / 4;
        for (int n = first; n < last; n++) {
          X[n] = (k & 1) ? -1 : 1;
        }
      }
    }
    for (int i = 0; i < (int)(dt / real_dt); i++) {
      if (i > 2) {
        for (int j = 0; j < N; j++) {
          X[j] = 0;
        }
      }
      integrate(X, real_dt);
      checkfired();
      Z1.push_back(CalcZ1());
      Z1.erase(Z1.begin());
    }

    nn = countfired();

    /*
    complex<double> *R = dft(Z1);
    R1 = abs(R[1]) / N;
    delete [] R;
    */
    double tmp = abs(Z1.back());
    dR1 = tmp - R1;
    R1 = tmp;

    for (int k = 0; k < 4; k++) {
      if ((a & (1 << k)) != 0) {
        tlast[k] = timestep * dt;
      }
    }
    timestep++;

    OneStepResult r(s, getReward(s, a));
    return r;
  }

  complex<double> CalcZ1() {
    complex<double> z1 = 0;

    for (int k = 0; k < 4; k++) {
      Zn[k] = 0;
      int first = (N * k)/4;
      int last = (N * (k + 1)) / 4;
      for (int j = first; j < last; j++) {
        Zn[k] += exp(i*1.0*y[j]);
      }
      Zn[k] /= N/4;
    }
    
    for (int j = 0; j < N; j++) {
      z1 += exp(i*1.0*y[j]);
    }
    return z1 / (double) N;
  }

  void integrate(int X[], double dt) {
    double yp[N];
    for (int j = 0; j < N; j++) {
      double s = 0;
      for (int k = 0; k < N; k++) {
        s += sin(y[j] - y[k]);
      }
      yp[j] = omega - K/N * s + X[j] * I * cos(y[j]) + D * rndNorm();
    }
    for (int j = 0; j < N; j++) {
      y[j] = y[j] + yp[j] * dt;
    }
    checkrange();
  }

  friend ostream & operator <<(ostream &os, TassModel m) {
    // Make sure this remains consistent with getValues()!!
    vector<double> state(m.nstate);
    m.getValues(&state[0]);
    for (int i = 0; i < m.nstate; i++) {
      os << state[i] << " ";
    }
    return os;
  }
};

/** Entry point to create a Tass object. */

Domain *getTass(const char *filename) {
  TassModel *tass = new TassModel();
  return tass;
}
