#include "rates.h"
#include "constString.h"
#include <cmath>
#include <cstdio> // debug

using namespace AprilUtils;

namespace Rates {

  typedef simple_matrix<char> matrix_path;

  class generic_rate {
    double alphas, alphai, alphab, alphaa;
    double betas,  betai,  betab,  betaa;
  public:
    generic_rate(double as, double ai, double ab, double aa,
                 double bs, double bi, double bb, double ba);
    double operator() (counter_edition);
  };

  generic_rate::generic_rate(double as, double ai, double ab, double aa,
			     double bs, double bi, double bb, double ba) :
    alphas(as), alphai(ai), alphab(ab), alphaa(aa),
    betas(bs),  betai(bi),  betab(bb),  betaa(ba) {
  }

  double generic_rate::operator() (counter_edition c) {
    return 100.0 * (alphas * c.ns + alphai * c.ni + alphab * c.nb + alphaa * c.na) /
      (betas * c.ns + betai * c.ni + betab * c.nb + betaa * c.na);
  }

  generic_rate rate_pra(0.0, 0.0, 0.0, 1.0,
                        1.0, 1.0, 1.0, 1.0);
  generic_rate rate_pre(1.0, 1.0, 1.0, 0.0,
                        1.0, 1.0, 1.0, 1.0);
  generic_rate rate_pa (0.0, 0.0, 0.0, 1.0,
                        1.0, 0.0, 1.0, 1.0);
  generic_rate rate_ip (0.0, -1.0, 0.0, 1.0,
                        1.0, 0.0, 1.0, 1.0);
  generic_rate rate_ie (1.0, 1.0, 1.0, 0.0,
                        1.0, 0.0, 1.0, 1.0);
  generic_rate rate_psb(1.0, 0.0, 1.0, 0.0,
                        1.0, 0.0, 1.0, 1.0);
  generic_rate rate_iep(1.0, 0.5, 0.5, 0.0,
                        1.0, 0.0, 1.0, 1.0);
  generic_rate rate_iap(0.0, -0.5, 0.5, 1.0,
                        1.0, 0.0, 1.0, 1.0);

  // funcion auxiliar para calcular la talla del diccionario
  int rates::dict_size(const pairs_int_sequences *data) {
    int resul = 0;
    for (const pairs_int_sequences *r = data; r != 0; r = r->next) {
      for (int i=0; i < r->correct.size; i++)
        if (resul < r->correct.symbol[i])
          resul = r->correct.symbol[i];
      for (int i=0; i < r->test.size; i++)
        if (resul < r->test.symbol[i])
          resul = r->test.symbol[i];
    }
    return resul;
  }

  struct dp_state {
    double score;
    counter_edition counter;
  };

  counter_edition rates::gp(double p,
                            int_sequence correct,
                            int_sequence test) {
    const double gs = 1.0;
    const double gi = p;
    const double gb = p;
    const double ga = 0.0;
    dp_state *previous = new dp_state[test.size+1];
    dp_state *current  = new dp_state[test.size+1];
    int itest, icorrect;
    current[0].score   = 0.0;
    current[0].counter = counter_edition(0,0,0,0);
    for (itest = 1; itest <= test.size; itest++) {
      // insertion
      current[itest] = current[itest-1];
      current[itest].score += gi;
      current[itest].counter.ni++;
    }
    for (icorrect = 0; icorrect < correct.size; icorrect++) {
      dp_state *swap = previous; previous = current; current = swap;
      current[0] = previous[0];
      current[0].score += gb;
      current[0].counter.nb++;
      for (itest = 1; itest <= test.size; itest++) {
        // deletion
        double score_b = previous[itest].score  + gb;
        // insertion
        double score_i = current[itest-1].score + gi;
        double score_min_ib = (score_i < score_b) ? score_i : score_b;
        // substitution
        double score_s = previous[itest-1].score +
          ((correct.symbol[icorrect] == test.symbol[itest-1]) ? ga : gs);
        //
        if (score_s <= score_min_ib) {
          current[itest].score = score_s;
          current[itest].counter = previous[itest-1].counter;
          if (correct.symbol[icorrect] == test.symbol[itest-1])
            current[itest].counter.na++;
          else
            current[itest].counter.ns++;
        } else {
          if (score_i < score_b) {
            current[itest].score = score_i;
            current[itest].counter = current[itest-1].counter;
            current[itest].counter.ni++;
          } else {
            current[itest].score = score_b;
            current[itest].counter = previous[itest].counter;
            current[itest].counter.nb++;
          }
        }
      }
    }
    counter_edition result = current[test.size].counter;
    delete[] previous;
    delete[] current;
    return result;
  }

  counter_edition rates::Gp(double p, 
                            const pairs_int_sequences *data) {
    counter_edition ce(0,0,0,0);
    for (const pairs_int_sequences *r = data; r != 0; r = r->next) {
      ce += gp(p, r->correct, r->test);
    }
    return ce;
  }

  struct dpn_state {
    double score;
    int len;
    counter_edition counter;
  };

  counter_edition rates::gp_normalized(int_sequence correct,
                                       int_sequence test) {
    const double gs = 1.0;
    const double gi = 1.0;
    const double gb = 1.0;
    const double ga = 0.0;
    dpn_state *previous = new dpn_state[test.size+1];
    dpn_state *current  = new dpn_state[test.size+1];
    int itest, icorrect;
    current[0].score   = 0.0;
    current[0].counter = counter_edition(0,0,0,0);
    current[0].len     = 1;
    for (itest = 1; itest <= test.size; itest++) {
      // insertion
      current[itest].score = 
        (current[itest-1].score+gi)/current[itest-1].len;
      current[itest].counter = current[itest-1].counter;
      current[itest].counter.ni++;
      current[itest].len = current[itest-1].len+1;
    }
    for (icorrect = 0; icorrect < correct.size; icorrect++) {
      dpn_state *swap = previous; previous = current; current = swap;
      current[0].score = (previous[0].score+gb)/previous[0].len;
      current[0].counter = previous[0].counter;
      current[0].counter.nb++;
      current[0].len = previous[0].len+1;
      for (itest = 1; itest <= test.size; itest++) {
        // deletion
        double score_b = 
          (previous[itest].score + gb)/previous[itest].len;
        // insertion
        double score_i = 
          (current[itest-1].score + gi)/current[itest-1].len;
        double score_min_ib = (score_i < score_b) ? score_i : score_b;
        // substitution
        double score_s = previous[itest-1].score +
          ((correct.symbol[icorrect] == test.symbol[itest-1]) ? ga : gs);
        score_s /= previous[itest-1].len;
        //
        if (score_s <= score_min_ib) {
          current[itest].score = score_s;
          current[itest].counter = previous[itest-1].counter;
          current[itest].len = previous[itest-1].len+1;
          if (correct.symbol[icorrect] == test.symbol[itest-1])
            current[itest].counter.na++;
          else
            current[itest].counter.ns++;
        } else {
          if (score_i < score_b) {
            current[itest].score = score_i;
            current[itest].counter = current[itest-1].counter;
            current[itest].counter.ni++;
            current[itest].len = current[itest-1].len+1;
          } else {
            current[itest].score = score_b;
            current[itest].counter = previous[itest].counter;
            current[itest].counter.nb++;
            current[itest].len = previous[itest].len+1;
          }
        }
      }
    }
    counter_edition result = current[test.size].counter;
    delete[] previous;
    delete[] current;
    return result;
  }

  double rates::initialize_lambda(const pairs_int_sequences *data) {
    counter_edition ce(0,0,0,0);
    for (const pairs_int_sequences *r = data; r != 0; r = r->next) {
      ce += gp_normalized(r->correct, r->test);
    }
    return (ce.ns + ce.ni + ce.nb) / 
      (double) (ce.ns + ce.ni + ce.nb + ce.na);
  }

  double rates::Fp(const pairs_int_sequences *data,
                   counter_edition &counted) {
    double lambda, lambdacero;
    double p;
    counter_edition ce;

    // programacion fraccional
    lambda = initialize_lambda(data);
    do {
      lambdacero = lambda;
      p = 1.0 - lambdacero / 2.0;
      ce = Gp (p,data);
      lambda = (ce.ns + ce.ni + ce.nb) / (double) (ce.ni + ce.nb + ce.ns + ce.na);
    } while (fabs (lambda - lambdacero) > 0.000001);

    counted = ce;
    return p;
  }

  // con recuperacion del camino
  counter_edition rates::gp_path(double p,
                                 int_sequence correct,
                                 int_sequence test,
                                 conf_matrix &confmat) {
    const double gs = 1.0;
    const double gi = p;
    const double gb = p;
    const double ga = 0.0;

    matrix_path mat(correct.size+1,test.size+1);
    char *m = mat.m;

    double *previous = new double[test.size+1];
    double *current  = new double[test.size+1];
    int itest, icorrect;
    current[0] = 0.0;
    *m = 's'; m++;
    for (itest = 1; itest <= test.size; itest++) {
      // insertion
      current[itest] = current[itest-1] + gi;
      //mat[0][itest] = 'i';
      *m = 'i'; m++;
    }
    for (icorrect = 1; icorrect <= correct.size; icorrect++) {
      double *swap = previous; previous = current; current = swap;
      current[0] = previous[0] +gb;
      *m = 'b'; m++;

      for (itest = 1; itest <= test.size; itest++) {
        // deletion
        double score_b = previous[itest] + gb;
        // insertion
        double score_i = current[itest-1] + gi;
        double score_min_ib = (score_i < score_b) ? score_i : score_b;
        // substitution
        double score_s = previous[itest-1] +
          ((correct.symbol[icorrect-1] == test.symbol[itest-1]) ? ga : gs);
        //
        if (score_s <= score_min_ib) {
          current[itest] = score_s;
          if (correct.symbol[icorrect-1] == test.symbol[itest-1]) {
            //mat[icorrect][itest] = 'a';
            *m = 'a'; m++;
          } else {
            //mat[icorrect][itest] = 's';
            *m = 's'; m++;
          }
        } else {
          if (score_i < score_b) {
            current[itest] = score_i;
            //mat[icorrect][itest] = 'i';
            *m = 'i'; m++;
          } else {
            current[itest] = score_b;
            //mat[icorrect][itest] = 'b';
            *m = 'b'; m++;
          }
        }
      }
    }
    //   fprintf(stderr,"----------------------------------------\n");
    //   printf(" correct: ");
    //   for (int i=0;i<correct.size;i++)
    //     printf("%d ",correct.symbol[i]);
    //   printf("\n test   : ");
    //   for (int i=0;i<test.size;i++)
    //     printf("%d ",test.symbol[i]);
    //   printf("\n");
    //   m = mat.m;
    //   for (int x=0;x<=correct.size;x++) {
    //     for (int y=0;y<=test.size;y++) {
    //       printf("%c",*m); m++;
    //     }
    //     printf("\n");
    //   }
  
    delete[] previous;
    delete[] current;
    // recuperar el camino y contabilizar matriz de confusion
    counter_edition result(0,0,0,0);
    icorrect = correct.size;
    itest    = test.size;
    while (icorrect > 0 || itest > 0) {

      //     fprintf(stderr,"icorrect=%d itest=%d %c %d %d\n",
      // 	    icorrect,itest,mat[icorrect][itest],
      // 	    correct.symbol[icorrect-1],test.symbol[itest-1]);

      switch (mat[icorrect][itest]) {
      case 'i':
        result.ni++;
        confmat[0][test.symbol[itest-1]]++;
        itest--;
        break;
      case 'b': 
        result.nb++;
        confmat[correct.symbol[icorrect-1]][0]++;
        icorrect--;
        break;
      case 's':
        result.ns++;
        confmat[correct.symbol[icorrect-1]][test.symbol[itest-1]]++;
        icorrect--;
        itest--;
        break;
      case 'a':
        result.na++; 
        confmat[correct.symbol[icorrect-1]][test.symbol[itest-1]]++;
        icorrect--;
        itest--;
        break;
      }
    }
    return result;
  }

  counter_edition rates::Gp_path(double p, 
                                 const pairs_int_sequences *data,
                                 conf_matrix &confmat) {
    counter_edition ce(0,0,0,0);
    for (const pairs_int_sequences *r = data; r != 0; r = r->next) {
      ce += gp_path(p, r->correct, r->test,confmat);
    }
    return ce;
  }

  double rates::Fp_path (const pairs_int_sequences *data,
                         counter_edition &counted,
                         conf_matrix &confmat) {
    double lambda, lambdacero;
    double p;
    counter_edition ce;

    // programacion fraccional
    lambda = initialize_lambda(data);
    do {
      lambdacero = lambda;
      p = 1.0 - lambdacero / 2.0;
      confmat.fill(0);
      ce = Gp_path(p,data,confmat);
      lambda = (ce.ns + ce.ni + ce.nb) / (double) (ce.ni + ce.nb + ce.ns + ce.na);
    } while (fabs (lambda - lambdacero) > 0.000001);

    counted = ce;
    return p;
  }

  // rate devuelve un vector con los tipos de rate
  double rates::rate(const pairs_int_sequences *data,
                     const char *rate_type,
                     counter_edition &counted,
                     double &p,
                     bool with_p,
                     conf_matrix **m) {

    constString cstipo = constString(rate_type);
    bool normalized = false;
    if (!with_p) {
      if (cstipo == "pra" ||
          cstipo == "pre") {
        normalized = true;
      } else {
        if (cstipo == "pa" ||
            cstipo == "psb" ||
            cstipo == "iep" ||
            cstipo == "iap")
          p = 0.5;
        else // ip, ie
          p = 1.0;
      }
    }
    if (m != 0) {
      int dsize = dict_size(data);
      conf_matrix *confmat = new conf_matrix(dsize+1,dsize+1);
      confmat->fill(0);
      if (normalized) {
        p = Fp_path(data,counted,*confmat);
      } else {
        counted = Gp_path(p,data,*confmat);
      }
      // tratar o guardar la matriz de confusion
      *m = confmat;
    } else {
      if (normalized) {
        p = Fp(data,counted);
      } else {
        counted = Gp(p,data);
      }
    }
    double rrate;
    if      (cstipo == "pra") {
      // pra -> porcentaje real de aciertos
      rrate = rate_pra(counted);
    }
    else if (cstipo == "pre") {
      // pre -> porcentaje real de errores
      rrate = rate_pre(counted);
    }
    else if (cstipo == "pa") {
      // pa  -> porcentaje de aciertos
      rrate = rate_pa(counted);
    }
    else if (cstipo == "ip") {
      // ip  -> indice de precision
      rrate = rate_ip(counted);
    }
    else if (cstipo == "ie") {
      // ie  -> indice de error
      rrate = rate_ie(counted);
    }
    else if (cstipo == "psb") {
      // psb -> porcentaje de sustituciones y borrados
      rrate = rate_psb(counted);
    }
    else if (cstipo == "iep") {
      // iep -> indice de error ponderado
      rrate = rate_iep(counted);
    }
    else if (cstipo == "iap") {
      // iap -> indice de acierto ponderado
      rrate = rate_iap(counted);
    }
    else {
      // TODO: REPORT ERROR???
      rrate = -1;
    }
    return rrate;
  }

  // void rates::debug(const pairs_int_sequences *data) {
  //   int ipair = 1;
  //   for (const pairs_int_sequences *r = data; r != 0; r = r->next) {
  //     printf("Pair %d\n",ipair); ipair++;
  //     printf(" correct: ");
  //     for (int i=0;i<r->correct.size;i++)
  //       printf("%d ",r->correct.symbol[i]);
  //     printf("\n test   : ");
  //     for (int i=0;i<r->test.size;i++)
  //       printf("%d ",r->test.symbol[i]);
  //     printf("\n");
  //   }
  // }

} // namespace Rates
