#include <cstdio>

#define EPSILON 1e-20f
#define MULT    1.01f //2.0f

int main() {
  for (float target=EPSILON; target <= 1.0f; target *= MULT) {
    for (float output=EPSILON; output <= 1.0f; output *= MULT) {
      float numerator   = output - target;
      float denominator = output * (1.0f - output);
      float frac   = numerator/denominator;
      float result = frac * denominator;
      printf ("%g %g => %g - %g = %g \n", output, target, numerator, result,
	      result-numerator);
    }
  }
  return 0;
}
