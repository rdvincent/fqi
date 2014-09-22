double rnd1() 
{
  return (double) rand() / RAND_MAX;
}

/**
 * Choose a random value on the given real interval.
 */
double rndInterval(double minv, double maxv)
{
  return rnd1() * (maxv - minv) + minv;
}


