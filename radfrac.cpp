/** 
 * \file Radiation fractionation model.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <map>

using namespace std;

#include "domain.h"
#include "random.h"

extern ostream & operator <<(ostream &os, const vector<double> &d);

class RFState {
  double N;
  double Gamma;
  friend class RadFrac;
};

class RFParam {
  double kN;                    // regrowth rate
  double gamma;                 // Scheidegger's gamma
  double alpha;
  double beta;
  int Tk;                       // time delay before regrowth
  double K;                     // 'carrying capacity'

  RFParam() {
    kN = 0;
    gamma = 0;
    alpha = 0;
    beta = 0;
    Tk = 0;
    K = 0;
  }

  friend ostream & operator <<(ostream &os, RFParam p) {
    os << p.alpha << " ";
    os << p.beta << " ";
    os << p.kN << " ";
    os << p.gamma << " ";
    os << p.Tk << " ";
    os << p.K;
    return os;
  }

  friend class RadFrac;
};

class RadFrac: public Domain {
private:
  double R0;                    // dose rate
  double N0;                    // initial cell count
  int NF;                       // total fractions
  int F;
  double T;                     // time per fraction, seconds
  RFState tumorState;
  RFState normalState;
  double D;                     /* total dosage */
  int rewardType;               /* reward function to use */
  double Amax;
  double *Amap;
  RFParam normal;
  RFParam tumor;

  /**
   * Return the current state from the domain object.
   *
   * We need this function since this domain commits the "sin" of 
   * incorporating the state into the object itself. Most of our
   * other domains avoid this. Need to think about how and if to
   * make this design better.
   */
  vector<double> getState() const {
    vector<double> s;
    s.resize(numDimensions);
    switch (numDimensions) {
    case 1:
      s[0] = F;
      break;
    case 2:
      s[0] = tumorState.N;
      s[1] = normalState.N;
      break;
    case 3:
      s[0] = tumorState.N;
      s[1] = normalState.N;
      s[2] = D;
      break;
    default:
      s[0] = tumorState.N;
      s[1] = normalState.N;
      s[2] = F;
      s[3] = D;
      break;
    }
    return s;
  }

public:
  RadFrac() {
    Amax = 1.0;
    numActions = 11;
    numDimensions = 4;
    numSteps = 1000;
    
    Amap = new double[numActions];
    for (int i = 0; i < numActions; i++) {
      Amap[i] = i * (Amax / (numActions - 1));
    }

    R0 = 0.64/60;               // convert from Gy/min to Gy/sec
    N0 = 1.0e11;                // initial number of cells
    NF = 4;                     // total number of fractions
    T = 1*24*3600;              // one day per fraction
    rewardType = 3;             // Default reward type.

    tumor.alpha = 1.43;        /* HT144 melanoma - Chapman */
    tumor.beta = 0.13;
    tumor.kN = 0.15/(3600*24);        // convert from 1/day to 1/sec
    tumor.gamma = 40.0/(3600*24);     // convert from 1/day to 1/sec
    tumor.Tk = 1;
    tumor.K = 2*N0;

    normal.alpha = 0.15;        /* Fibrosis - Bentzen et al 1990 */
    normal.beta = 0.079;
    normal.kN = 0.15/(3600*24);        // convert from 1/day to 1/sec
    normal.gamma = 71.0/(3600*24);     // convert from 1/day to 1/sec
    normal.Tk = 0;
    normal.K = N0;

    reset();
  }

  // Just resets the state to its initial conditions.
  void reset() {
    F = 0;
    normalState.N = tumorState.N = N0;
    normalState.Gamma = tumorState.Gamma = 0;
    D = 0;
  }

  void setFeatures(int n) { numDimensions = n; }

  void initRandomly() { tumorState.N = N0 + (rndNorm() * N0/50.0); }

  void loadProperties(const char *fname) {
    if (fname == NULL) {
      return;
    }
    ifstream in;
    map<string,double> props;

    in.open(fname);
    if (in.fail()) {
      cerr << "Can't open properties file: " << fname << endl;
      exit(-1);
    }
    else { 
      string line;
      while (getline(in, line)) {
        string key;
        double val;
        if (line[0] != '#') {
          istringstream iss(line);
          if (!(iss >> key >> val)) {
            if (key.length() > 0) {
              cerr << key << "->" << val << endl;
              cerr << "Problem reading properties file?" << endl;
            }
            break;
          }
          if (key.length() > 0) {
            props[key] = val;
          }
        }
      }
      in.close();
    }

    if (props.count("dose-rate") > 0) {
      R0 = props.at("dose-rate") / 60;
    }
    if (props.count("reward-type") > 0) {
      rewardType = (int)props.at("reward-type");
    }
    if (props.count("fraction-interval") > 0) {
      T = (int)(props.at("fraction-interval")*24*3600);
    }
    if (props.count("fraction-count") > 0) {
      NF = (int)props.at("fraction-count");
    }
    if (props.count("tumor-alpha-fraction") > 0) {
      tumor.alpha *= props.at("tumor-alpha-fraction");
    }
    if (props.count("tumor-beta-fraction") > 0) {
      tumor.beta *= props.at("tumor-beta-fraction");
    }
    if (props.count("tumor-alpha") > 0) {
      tumor.alpha = props.at("tumor-alpha");
    }
    if (props.count("tumor-beta") > 0) {
      tumor.beta = props.at("tumor-beta");
    }
    if (props.count("normal-alpha") > 0) {
      normal.alpha = props.at("normal-alpha");
    }
    if (props.count("normal-beta") > 0) {
      normal.beta = props.at("normal-beta");
    }
    if (props.count("tumor-regrowth-rate") > 0) {
      tumor.kN = props.at("tumor-regrowth-rate") / (3600 * 24);
    }
    if (props.count("normal-regrowth-rate") > 0) {
      normal.kN = props.at("normal-regrowth-rate") / (3600 * 24);
    }
    if (props.count("tumor-gamma") > 0) {
      tumor.gamma = props.at("tumor-gamma") / (3600*24);
    }
    if (props.count("normal-gamma") > 0) {
      normal.gamma = props.at("normal-gamma") / (3600*24);
    }
    if (props.count("tumor-regrowth-delay") > 0) {
      tumor.Tk = props.at("tumor-regrowth-delay") * 3600 * 24;   /* convert from days to seconds */
    }
    if (props.count("tumor-k") > 0) {
      tumor.K = props.at("tumor-k") * N0;
    }
    if (props.count("normal-k") > 0) {
      normal.K = props.at("normal-k") * N0;
    }
    if (props.count("features") > 0) {
      numDimensions = props.at("features");
    }
  }

  int fractions() {
    return NF;
  }

  void setFractions(int n) {
    NF = n;
  }

  vector<double> initialState() {
    vector<double> s;
    s.resize(numDimensions);
    // Need to return internal state to initial conditions here.
    reset();

    s[0] = tumorState.N;
    s[1] = normalState.N;
    s[2] = F;
    s[3] = D;
    return s;
  }

  bool isTerminal(vector<double> s) const {
    return (s[2] >= NF);
  }

  double getReward(vector<double> s, int a) const {
    double normalRatio = normalState.N / N0;
    double tumorRatio = tumorState.N / N0;

    if (normalRatio > 1.0) {
      normalRatio = 1.0;
    }
    if (tumorRatio > 1.0) {
      tumorRatio = 1.0;
    }

    switch (rewardType) {
    case 1:
      return (F == NF) ? (normalRatio * normalRatio - tumorRatio) : 0.0;
    case 2:
      return (F == NF) ? (normalRatio - sqrt(tumorRatio)) : 0.0;
    case 3:
      return (normalRatio < 0.90) ? -1.0 : (F == NF) ? (1.0 - tumorRatio) : 0;
    case 11:
      return (normalRatio * normalRatio) - sqrt(tumorRatio);
    default:
      return (normalRatio < 0.90) ? -1.0 : (F == NF) ? (1.0 - sqrt(tumorRatio)) : 0.0;
    }
  }

  void scheidegger(const RFParam& param, double F, RFState& state) {
    double R = R0;
    double D = 0;
    double dt = 0.05;
    int X = (int)(1/dt);
    int delay = 0;

    if (F != 0.0) {
      delay = param.Tk;
    }
    
    for (int t = 0; t < T; t++) {
      // Integrate one second...
      for (int i = 0; i < X; i++) {
        // Check if we have passed the maximum dose.
        if (D >= F) {
          R = 0;
        }
        
        /* This is the equation for "second order" kinetics, we
         * need to change if we want to do "first order" models.
         * It's implied by equations 3 and 8 in the paper.
         */
        double dGamma = R - param.gamma * state.Gamma * state.Gamma;
        double dN;
        /* This is equation 6 from the paper
         */
        if (delay <= 0) {
          if (param.K > 0.0)
            dN = -(param.alpha + 2 * param.beta * state.Gamma) * R * state.N + param.kN * state.N * (1.0 - state.N / param.K);
          else
            dN = -(param.alpha + 2 * param.beta * state.Gamma) * R * state.N + param.kN * state.N;
        }
        else {
          /* Delay regrowth if specified.
           */
          dN = -(param.alpha + 2 * param.beta * state.Gamma) * R * state.N;
        }
            
        // Simple Euler integration of Gamma, N, and the dose
        state.Gamma = state.Gamma + dt * dGamma;
        state.N = state.N + dt * dN;
        D = D + dt * R;
      }

      /* Decrement the regrowth delay if appropriate. 
       */
      if (delay > 0) {
        delay--;
      }
    }
  }

  OneStepResult performAction(vector<double> s, int a) {
    double FD = Amap[a];

    // integrate tumor
    scheidegger(tumor, FD, tumorState);
    // integrate normal
    scheidegger(normal, FD, normalState);

    F += 1;
    D += FD;
    OneStepResult r(getState(), getReward(s, a));
    return r;
  }

  void printParameters(ostream &os) {
    os << "% " << NF << " " << T << " " << R0 << endl;
    os << "% " << tumor << " " << normal << endl;
  }

  friend ostream & operator <<(ostream &os, RadFrac m) {
    os << m.getState() << endl;
    return os;
  }
};

// public entry point to construct a RadFrac object.

Domain *getRF(const char *filename) {
  RadFrac *rf = new RadFrac();
  rf->loadProperties(filename);
  return rf;
}
