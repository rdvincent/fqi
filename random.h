/**
 * \file
 * \brief Random number generation.
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
/**
 * Return a uniform random number between 0 and 1. Uses the library
 * rand() function, which is probably a bad idea.
 * \return A double-precision floating point random number.
 */
double rnd1() 
{
  return (double) rand() / RAND_MAX;
}

/**
 * Choose a random value on the given real interval.
 * \param minv Minimum value.
 * \param maxv Maximum value.
 * \return The random number.
 */
double rndInterval(double minv, double maxv)
{
  return rnd1() * (maxv - minv) + minv;
}

/**
 * Gaussion random variable with zero mean and variance 1.0, using
 * box-muller transform. Not thread safe!
 */
double rndNorm() {
  static int iset = 0;
  static double gset;
  double fac, rsq, v1, v2;

  if (iset == 0) {
    do {
      v1 = 2.0 * rnd1() - 1.0;
      v2 = 2.0 * rnd1() - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  }
  else {
    iset = 0;
    return gset;
  }
}
