#include <vector>
#include <utility>
#include <cmath>
#include <string.h>

using namespace std;

#include "domain.h"

extern double rndInterval(double, double);

double sign(double x) {
  if (x == 0.0) return 0.0;
  if (x < 0.0) return -1.0;
  return 1.0;
}

/*
double dangle(double x) {
  return fabs(x + 2.0*k*M_PI);
}
*/

class Bicycle : public Domain {
private:
  double dt;
  double v;
  double g;
  double d_CM;
  double c;
  double h;
  double M_c;
  double M_d;
  double M_p;
  double M;
  double r;
  double l;

  //double delta_psi;             // used for reward calculation.
public:
  Bicycle() {
    numActions = 9;
    numSteps = 500;
    numDimensions = 7;

    dt = 0.01;
    v = 10.0/3.6;
    g = 9.82;
    d_CM = 0.3;
    c = 0.66;
    h = 0.94;
    M_c = 15.0;
    M_d = 1.7;
    M_p = 60.0;
    M = M_c + M_p;
    r = 0.34;
    l = 1.11;
  }

  bool isTerminal(vector<double> s) {
    return (s[0] > M_PI*12.0/180.0);
  }

  double getReward(vector<double> s, int a) {
    if (isTerminal(s)) return -1.0;
    //return 0.1 * delta_psi;
    return 0;
  }
  
  pair<vector<double>,double> performAction(vector<double> s, int a) {
    vector<double> sp(numDimensions, 0.0);

    double ad[] = { 0.0, 0.0,  0.0, -0.02, -0.02, -0.02, 0.02, 0.02, 0.02 };
    double aT[] = { 0.0, 2.0, -2.0,  0.0,   2.0,  -2.0,  0.0,  2.0, -2.0 };
    double d = ad[a];
    double T = aT[a];
    double w = rndInterval(-0.02, 0.02);

    double dot_sigma = v/r;
    double I_bnc = 13.0/3.0*M_c*h*h + M_p*(h + d_CM)*(h + d_CM);
    double I_dc = M_d * r * r;
    double I_dv = 3.0/2.0 * M_d * r * r;
    double I_dl = 1.0/2.0 * M_d * r * r;

    double omega = s[0];
    double dot_omega = s[1];
    double theta = s[2];
    double dot_theta = s[3];
    double x_b = s[4];
    double y_b = s[5];
    double psi = s[6];

    double phi = omega + atan(d + w)/h;
    double invrf = fabs(sin(theta))/l;
    double invrb = fabs(tan(theta))/l;
    double invrcm = (theta == 0.0) ? 0.0 : 1.0/sqrt((l-c)*(l-c) + (1.0/invrb)*(1.0/invrb));

    sp[0] = omega + dt * dot_omega;
    sp[1] = dot_omega + dt * (1.0 / I_bnc) * (M*h*g*sin(phi) - cos(phi)*(I_dc*dot_sigma*dot_theta + sign(theta)*v*v*(M_d*r*(invrb+invrf)+M*h*invrcm)));
    sp[2] = theta + dt * dot_theta;
    sp[3] = dot_theta + dt * (T - I_dv*dot_sigma*dot_omega)/I_dl;
    sp[4] = x_b + dt * v * cos(psi);
    sp[5] = y_b + dt * v * sin(psi);
    sp[6] = psi + dt * sign(theta)*v*invrb;

    //delta_psi = dangle(psi) - dangle(sp[6]);

    if (fabs(theta) > M_PI*80.0/180.0) {
      sp[2] = sign(theta)*M_PI*80.0/180.0;
      sp[3] = 0.0;
    }

    pair<vector<double>, double> p(sp, getReward(sp, a));
    return p;
  }

  vector<double> initialState() {
    vector<double> s(numDimensions, 0.0);

    s[6] = M_PI;
    return s;
  }
};

/*! Defines the HIV problem as defined by Adams et al. and as used by
 * Ernst et al. 2006
 */
class HIV : public Domain {
private:
  double Q;
  double R1;
  double R2;
  double S;

  double l1;              // type 1 cell production rate
  double d1;                 // type 1 cell death rate
  double k1;               // population 1 infection rate
  double l2;                // type 2 cell production rate
  double d2;                 // type 2 cell death rate
  double f; // treatment efficacy reduction in population 2
  double k2;                // population 2 infection rate 
  double delta;              // infected cell death rate
  double m1; // immune-induced clearance rate for population 1
  double m2; // immune-induced clearance rate for population 2
  double NT;    // virions produced per infected cell
  double c;      // virus natural death rate
  double p1; // average number of virions infecting a type 1 cell
  double p2; // average number of virions infecting a type 2 cell
  double lE; // immune effector production rate
  double bE; // maximum birth rate for immune effectors
  double Kb; // saturation constant for immune effector birth
  double dE;  // maximum death rate for immune effectors
  double Kd; // saturation constant for immune effector death
  double deltaE; // natural death rate for immune effectors

  // Other constants
  double dt;                // Our integration timestep, in days.
  int nInt;                 // Number of integration steps per action.

public:
  HIV() {
    numActions = 4;
    numSteps = 200;

    numDimensions = 6;
    // constants for the reward function
    Q = 0.1;
    R1 = 20000.0;
    R2 = 2000.0;
    S = 1000.0;

    // Constants for the ODE's
    l1 = 10000.0;              // type 1 cell production rate
    d1 = 0.01;                 // type 1 cell death rate
    k1 = 8.0e-7;               // population 1 infection rate
    l2 = 31.98;                // type 2 cell production rate
    d2 = 0.01;                 // type 2 cell death rate
    f = 0.34; // treatment efficacy reduction in population 2
    k2 = 1e-4;                // population 2 infection rate 
    delta = 0.7;              // infected cell death rate
    m1 = 1.0e-5; // immune-induced clearance rate for population 1
    m2 = 1.0e-5; // immune-induced clearance rate for population 2
    NT = 100.0;    // virions produced per infected cell
    c = 13.0;      // virus natural death rate
    p1 = 1.0; // average number of virions infecting a type 1 cell
    p2 = 1.0; // average number of virions infecting a type 2 cell
    lE = 1.0; // immune effector production rate
    bE = 0.3; // maximum birth rate for immune effectors
    Kb = 100.0; // saturation constant for immune effector birth
    dE = 0.25;  // maximum death rate for immune effectors
    Kd = 500.0; // saturation constant for immune effector death
    deltaE = 0.1;           // natural death rate for immune effectors

    // Other constants
    dt = 0.001;               // Our integration timestep, in days.
    nInt = (int)(5.0 / dt);   // Number of integration steps per action.
  }


  double getReward(vector<double> s, int a) {
    // e1 is between 0.0 and 0.7 (RTI therapy on/off)
    // e2 is between 0.0 and 0.3 (PI therapy on/off)

    double V = s[4];
    double E = s[5];

    double e1 = ((a & 1) != 0) ? 0.7 : 0.0;
    double e2 = ((a & 2) != 0) ? 0.3 : 0.0;

    return -(Q*V + R1*e1*e1 + R2*e2*e2 - S*E);
  }

  pair<vector<double>,double> performAction(vector<double> x, int a) {
    /* The state vector has the format T1,T2,T1*,T2*,V,E */

    /* This version is restricted to only four possible actions.
     */
    double e1 = ((a & 1) != 0) ? 0.7 : 0.0;
    double e2 = ((a & 2) != 0) ? 0.3 : 0.0;

    vector<double> dy(numDimensions);
    vector<double> y = x;

    for (int i = 0; i < nInt; i++) {
      dy[0] = l1 - d1 * y[0] - (1 - e1) * k1 * y[4] * y[0];
      dy[1] = l2 - d2 * y[1] - (1 - f * e1) * k2 * y[4] * y[1];
      dy[2] = (1 - e1) * k1 * y[4] * y[0] - delta * y[2] - m1 * y[5] * y[2];
      dy[3] = (1 - f * e1) * k2 * y[4] * y[1] - delta * y[3] - m2 * y[5] * y[3];
      dy[4] = (1.0 - e2) * NT * delta * (y[2] + y[3]) - c * y[4] - 
        ((1 - e1) * p1 * k1 * y[0] + (1 - f * e1) * p2 * k2 * y[1]) * y[4];
      dy[5] = lE + (bE * (y[2] + y[3]) * y[5]) / (y[2] + y[3] + Kb) -
        (dE * (y[2] + y[3]) * y[5]) / (y[2] + y[3] + Kd) - deltaE * y[5];

      for (int j = 0; j < numDimensions; j++)
        y[j] += dy[j] * dt;
    }

    pair<vector<double>, double> p(y, getReward(y, a));
    return p;
  }

  vector<double> initialState() {
    vector<double> s(numDimensions);

    /* This is the "sick" initial state.
     */
    s[0] = 163574.0;
    s[1] = 5.0;
    s[2] = 11945.0;
    s[3] = 46.0;
    s[4] = 63919.0;
    s[5] = 24.0;

    /* These states correspond to "infected but healthy" and "not infected".
     */
    // (967839.0, 621.0, 76.0, 6.0, 415.0, 353108.0)
    // (1000000.0, 3198.0, 0.0, 0.0, 0.0, 10.0)
    return s;
  }
};

/*! Implementation of the classic "mountain-car" domain from Singh and Sutton 1996.
 */
class MC : public Domain {
public:
  MC() {
    numDimensions = 2;
    numActions = 3;
    numSteps = 2000;
  }

  bool isStochastic() { return true; }

  double getReward(vector<double> s, int a) {
    if (isTerminal(s)) { 
      return 0.0;
    }
    else return -1.0;
  }

  vector<double> initialState() {
    vector<double> s(numDimensions);
    s[0] = rndInterval(-1.2, 0.5);
    s[1] = rndInterval(-0.07, 0.07);
    return s;
  }

  pair<vector<double>,double> performAction(vector<double> s, int a) {
    double acc = 0.0;
    if (a == 0) {
      acc = -1.0;
    }
    if (a == 2) {
      acc = 1.0;
    }
  
    double x0 = s[0];
    double v0 = s[1];
    double v1 = v0 + acc * 0.001 + cos(3.0 * x0) * -0.0025;

    // Enforce bounds.
    if (v1 < -0.07) {
      v1 = -0.07;
    }
    else if (v1 > 0.07) {
      v1 = 0.07;
    }
    double x1 = x0 + v1;
    if (x1 < -1.2) {
      x1 = -1.2;
    }
    else if (x1 > 0.5) {
      x1 = 0.5;
    }

    vector<double> s1(numDimensions);
    s1[0] = x1;
    s1[1] = v1;
    pair<vector<double>,double> p(s1, getReward(s1, a));
    return p;
  }

  bool isTerminal(vector<double> s) { return (s[0] >= 0.5); }
};

/*! Create a domain by name. Avoids having to export domain classes outside 
 * this module.
 */
Domain *CreateDomain(const char *name) {
  if (!strcasecmp(name, "MC")) {
    return new MC();
  }
  else if (!strcasecmp(name, "Bicycle")) {
    return new Bicycle();
  }
  else {
    return new HIV();
  }
}

