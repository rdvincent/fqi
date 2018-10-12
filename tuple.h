/**
 * \file
 * \brief An entry in the input FQI training set.
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
/**
 * The class that represents an entry in the raw training set. Each tuple
 * represents an instance of the one-step dynamics of the RL environment.
 */
class fqi_tuple {
public:
  vector<double> s;             /*!< Initial state. */
  int a;                        /*!< Action */
  double r;                     /*!< Reward */
  vector<double> sp;            /*!< Next state. */

  /** Create a new instance of a fqi_tuple.
   */
  fqi_tuple(vector<double> s, int a, double r, vector<double> sp) {
    this->s = s;
    this->a = a;
    this->r = r;
    this->sp = sp;
  }
};


