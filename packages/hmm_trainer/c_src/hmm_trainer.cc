/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Jorge Gorbe Moya, Salvador España-Boquera
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
#include "error_print.h"
#include "hmm_trainer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib> // para exit
#include <cstring>
#include "april_assert.h"

using namespace AprilUtils;
using namespace Basics;

namespace HMMs {

  hmm_trainer::hmm_trainer() {

    num_cls_states      = 0;
    num_cls_transitions = 0;
    num_cls_emissions   = 0;
    vsz_cls_states      = 4;
    vsz_cls_transitions = 4;
    vsz_cls_emissions   = 4;
    cls_transition = new hmm_trainer_cls_transition[vsz_cls_transitions];
    cls_state      = new hmm_trainer_cls_state[vsz_cls_states];
    apriori_cls_emission = new log_float[vsz_cls_emissions];
    acum_cls_emission    = new log_double[vsz_cls_emissions];

    // inicializar aprioris, como si no hubiese el calculo de aprioris:
    for (int i=0; i<num_cls_emissions; i++)
      apriori_cls_emission[i] = log_float::one();

  }

  hmm_trainer::~hmm_trainer() {
    delete[] cls_transition;
    delete[] cls_state;
    delete[] apriori_cls_emission;
    delete[] acum_cls_emission;
  }

  void hmm_trainer::check_cls_state(int st) {
    if (st >= vsz_cls_states) {
      while (st >= vsz_cls_states) 
        vsz_cls_states *= 2;
      hmm_trainer_cls_state *old_cls_state = cls_state;
      cls_state = new hmm_trainer_cls_state[vsz_cls_states];
      for (int i=0; i<num_cls_states; i++)
        cls_state[i] = old_cls_state[i];
      delete[] old_cls_state;
    }
    if (st >= num_cls_states)
      num_cls_states = st+1;
  }

  int hmm_trainer::new_cls_state() {
    int aux = num_cls_states;
    check_cls_state(aux);
    cls_state[aux].cls_tr_list = -1; // lista vacia
    return aux;
  }

  void hmm_trainer::check_cls_transition(int tr) {
    if (tr >= vsz_cls_transitions) {
      while (tr >= vsz_cls_transitions) 
        vsz_cls_transitions *= 2;
      hmm_trainer_cls_transition *old_cls_transition = cls_transition;
      cls_transition = new hmm_trainer_cls_transition[vsz_cls_transitions];
      for (int i=0; i<num_cls_transitions; i++)
        cls_transition[i] = old_cls_transition[i];
      delete[] old_cls_transition;
    }
    if (tr >= num_cls_transitions)
      num_cls_transitions = tr+1;
  }

  int hmm_trainer::new_cls_transition(int c_st) {
    int aux = num_cls_transitions;
    check_cls_transition(aux);
    cls_transition[aux].prob = log_float::zero();
    cls_transition[aux].emission = 0; // cualquier valor >=0
    if (c_st == -1)
      cls_transition[aux].next = -1;
    else {
      cls_transition[aux].next = cls_state[c_st].cls_tr_list;
      cls_state[c_st].cls_tr_list = aux;
    }
    return aux;
  }

  void hmm_trainer::check_cls_emission(int emis) {
    if (emis >= vsz_cls_emissions) {
      while (emis >= vsz_cls_emissions) 
        vsz_cls_emissions *= 2;
      delete[] apriori_cls_emission;
      delete[] acum_cls_emission;
      apriori_cls_emission = new log_float[vsz_cls_emissions];
      acum_cls_emission    = new log_double[vsz_cls_emissions];
      // inicializar aprioris, como si no hubiese el cÃ¡lculo de
      // aprioris:
      for (int i=0; i<num_cls_emissions; i++)
        apriori_cls_emission[i] = log_float::one();
    }
    if (emis >= num_cls_emissions) {
      for (int i=num_cls_emissions; i<=emis; i++)
        apriori_cls_emission[i] = log_float::one();
      num_cls_emissions = emis+1;
    }
  }

  void hmm_trainer::begin_expectation() {
    for (int i=0;i<num_cls_transitions;i++)
      cls_transition[i].acum = log_double::zero();
    for (int i=0;i<num_cls_emissions;i++)
      acum_cls_emission[i] = log_double::zero();
  }

  void hmm_trainer::end_expectation(bool update_trans_prob, 
                                    bool update_a_priori_emission) {
    log_double maxpracum;
    log_double sumatotal;
    int rt;
    if (update_trans_prob) {
#ifdef CHECK_HMMTRAINER_TRANS_PROB_DIFERENT_ZERO
      log_float minprob = log_float::one(),auxprob;
      // comprobamos que no haya probabilidades a cero
      for (int i=0;i<num_cls_states;i++) {
        rt=cls_state[i].cls_tr_list;
        if (rt != -1) {
	
          // DEBUG
          for (rt = rt=cls_state[i].cls_tr_list;
               rt != -1;
               rt = cls_transition[rt].next)
            printf("cls_transition[%d] emission %2d acum %f\n",
                   rt,cls_transition[rt].emission,
                   cls_transition[rt].acum.to_double());
          rt=cls_state[i].cls_tr_list;
          // FIN DEBUG

          maxpracum = cls_transition[rt].acum;
          for (rt = cls_transition[rt].next; 
               rt != -1;
               rt = cls_transition[rt].next)
            if (maxpracum < cls_transition[rt].acum)
              maxpracum = cls_transition[rt].acum;
	
          // calcula la suma total:
          rt = cls_state[i].cls_tr_list;
          cls_transition[rt].acum /= maxpracum;
          sumatotal = cls_transition[rt].acum;
          for (rt = cls_transition[rt].next;
               rt != -1;
               rt = cls_transition[rt].next) {
            cls_transition[rt].acum /= maxpracum;
            sumatotal += cls_transition[rt].acum;
          }
          for (rt = cls_state[i].cls_tr_list;
               rt != -1;
               rt = cls_transition[rt].next) {
	  
            printf("cls_tr[%3d] = %f -> %f = %f/%f\n",
                   rt,
                   cls_transition[rt].prob.to_double(),
                   (cls_transition[rt].acum / sumatotal).to_double(),
                   cls_transition[rt].acum.log(), sumatotal.log());
	  
            auxprob = cls_transition[rt].acum / sumatotal;
            if (auxprob < minprob)
              minprob = auxprob;
          }
        }
      }
      if (minprob > log_float::zero()) {
#endif
        // recalcular las probabilidades de las transiciones
        for (int i=0;i<num_cls_states;i++) {
          rt=cls_state[i].cls_tr_list;
          if (rt != -1) {
            maxpracum = cls_transition[rt].acum;
            for (rt = cls_transition[rt].next; 
                 rt != -1;
                 rt = cls_transition[rt].next)
              if (maxpracum < cls_transition[rt].acum)
                maxpracum = cls_transition[rt].acum;
	  
            // calcula la suma total:
            rt = cls_state[i].cls_tr_list;
            cls_transition[rt].acum /= maxpracum;
            sumatotal = cls_transition[rt].acum;
            for (rt = cls_transition[rt].next;
                 rt != -1;
                 rt = cls_transition[rt].next) {
              cls_transition[rt].acum /= maxpracum;
              sumatotal += cls_transition[rt].acum;
            }
            for (rt = cls_state[i].cls_tr_list;
                 rt != -1;
                 rt = cls_transition[rt].next) {
	    
              // 	printf("cls_tr[%3d] = %f -> %f = exp(%f)/exp(%f)\n",
              // 	       rt, cls_transition[rt].prob.to_float(),
              // 	       (cls_transition[rt].acum / sumatotal).to_float(),
              // 	       cls_transition[rt].acum.log(), sumatotal.log());
	    
              cls_transition[rt].prob = cls_transition[rt].acum / sumatotal;
            }
          }
        }
#ifdef CHECK_HMMTRAINER_TRANS_PROB_DIFERENT_ZERO
      } // if minprob > 0.0
#endif
    } // if update...
    if (update_a_priori_emission) {
    
      // recalcular las probabilidades a priori de los tipos de emisión
      maxpracum = acum_cls_emission[0];
      for (int i=1;i<num_cls_emissions;i++)
        if (maxpracum < acum_cls_emission[i])
          maxpracum = acum_cls_emission[i];
    
      acum_cls_emission[0] /= maxpracum;
      sumatotal = acum_cls_emission[0];
      for (int i=1;i<num_cls_emissions;i++) {
        acum_cls_emission[i] /= maxpracum;
        sumatotal += acum_cls_emission[i];
      }
      for (int i=0;i<num_cls_emissions;i++) {
      
        //     printf("apriori[%d] = %f -> %f = exp(%f)/exp(%f)\n",
        // 	   i, apriori_cls_emission[i].to_float(),
        // 	   (acum_cls_emission[i] / sumatotal).to_float(),
        // 	   acum_cls_emission[i].log(),sumatotal.log());
      
        apriori_cls_emission[i] = acum_cls_emission[i] / sumatotal;
        // PARCHE FEO HEURISTICO: <- IMPORTANTE
        if (apriori_cls_emission[i] <= log_float::from_float(1e-20))
          apriori_cls_emission[i] = log_float::from_float(1e20);
      }
    }
  }

  void hmm_trainer::print() const {
  }

  hmm_trainer_model::hmm_trainer_model(hmm_trainer *the_trainer) {

    created         = false;
    num_states      = 0;
    num_transitions = 0;
    initial_state   =-1;
    final_state     =-1;
    list_transitions= 0;
    transition      = 0;
    //ranking         = 0;
    trainer         = the_trainer;
    IncRef(trainer);
  }

  int hmm_trainer_model::new_state() {
    int aux = num_states;
    num_states++;
    return aux;
  }

  void hmm_trainer_model::new_transition(int from, int to, 
                                         int emission, 
                                         int cls_transition,
                                         log_float prob,
                                         const char *output) {
    num_transitions++;
    hmm_aux_transition *aux = new hmm_aux_transition;
    aux->from     = from;
    aux->to       = to;
    aux->emission = emission;
    trainer->check_cls_emission(emission);
    aux->id       = cls_transition;
    aux->prob     = prob;
    aux->next     = list_transitions;
    list_transitions = aux;
    if (output == 0)
      aux->output = 0;
    else {
      aux->output = new char[strlen(output)+1];
      strcpy(aux->output,output);
    }
  }

  // clase auxiliar para calcular el orden topologico:
  struct list_top_order {
    int state;
    list_top_order *next;
    list_top_order(int st, list_top_order *nxt) : state(st), next(nxt) {}
  };

  bool hmm_trainer_model::prepare_model() { // llamarlo una vez introducido todo
    if (num_transitions <= 0 || initial_state < 0 || final_state < 0) {
      ERROR_PRINT3("prepare_model(): num_transitions(%d) <= 0 || "
                   "initial_state(%d) < 0 || final_state(%d) < 0\n",
                   num_transitions, initial_state, final_state);
      return false;
    }

    // calcular orden de los estados que induce un orden sobre las
    // transiciones para que las de tipo lambda funcionen correctamente:
    transition = new hmm_trainer_transition[num_transitions];

    // lista para guardar, para cada estado, las transiciones
    // que salen de ese estado:
    hmm_aux_transition **salen_de = new hmm_aux_transition*[num_states];

    // numlambdas cuenta num tr. lambda q llegan a cada estado
    int *numlambdas = new int[num_states];
    // ranking cuenta el camino de lambdas más largo que llega a cada
    // estado:
    // ranking         = new int[num_states];
    // first_transition= new int[num_states+1];
    for (int i=0;i<num_states;i++) {
      salen_de[i]   = 0;
      numlambdas[i] = 0;
      // ranking[i]    = 0;
    }
    // pasamos los nodos de la lista list_transitions a salen_de y
    // aprovechamos para contar numlambdas:
    while (list_transitions) {
      hmm_aux_transition *aux = list_transitions;;
      list_transitions = list_transitions->next;
      aux->next = salen_de[aux->from];
      salen_de[aux->from] = aux;
      if (aux->emission < 0) { // y si es de tipo lambda
        numlambdas[aux->to]++;
      }
    }

    // generamos una lista con los estados con 0 transiciones lambda en
    // su entrada:
    list_top_order *listop = 0;
    for (int i=0;i<num_states;i++)
      if (numlambdas[i] == 0) // anyadir el estado "i" a la lista de estados
        listop = new list_top_order(i,listop);
  
    // procesar los estados en orden topologico respecto al grafo
    // compuesto unicamente por las transiciones lambda
    int itr = 0;
    while (listop != 0) {
      list_top_order *a = listop;
      listop = listop->next;
      int st = a->state;
      delete a;
      // procesar todas las transiciones que salen de st
      while (salen_de[st] != 0) {
        int emis              = salen_de[st]->emission;
        int dest              = salen_de[st]->to;
        transition[itr].from  = salen_de[st]->from;
        transition[itr].to    = dest;
        transition[itr].output= salen_de[st]->output;
        transition[itr].cls_transition = salen_de[st]->id;
        trainer->set_cls_transition_emission(salen_de[st]->id,emis);
        trainer->set_cls_transition_prob(salen_de[st]->id,salen_de[st]->prob);
        itr++;
        if (emis < 0) { // si es transicion lambda:
          // if (ranking[dest] < ranking[st]+1) // actualizar ranking
          //   ranking[dest] = ranking[st]+1;   // estado destino
          numlambdas[dest]--; // meterlo en lista para orden topologico
          if (numlambdas[dest] == 0)  // si hace falta
            listop = new list_top_order(dest,listop);
        }
        // eliminar la transicion de la lista:
        hmm_aux_transition* aux = salen_de[st];
        salen_de[st] = salen_de[st]->next;
        delete aux;
      }
    } // while (listop != 0);

    delete[] salen_de;
    delete[] numlambdas;

    // comprobar que el grafo no tenia ciclos de tr. lambda itr deberia
    // valer num_transitions
    if (itr != num_transitions) {
      delete[] transition;       transition=0;
      // delete[] ranking;          ranking=0;
      //delete[] first_transition; first_transition=0;
      ERROR_PRINT2("prepare_model(): error, itr(%d) != num_transitions(%d)\n",
                   itr, num_transitions);
      return false;
    }
    created = true;
    return true;
  }

  hmm_trainer_model::~hmm_trainer_model() {
    DecRef(trainer);
    if (transition != 0) {
      for (int i=0; i<num_transitions; i++)
        if (transition[i].output != 0)
          delete[] transition[i].output;
      delete[] transition;
    }
  }

  inline log_float hmm_trainer_model::transition_prob(int tr) {
    int i = transition[tr].cls_transition;
    return trainer->get_cls_transition_prob(i);
  }

  inline int hmm_trainer_model::transition_emission(int tr) {
    int i = transition[tr].cls_transition;
    return trainer->get_cls_transition_emission(i);
  }

  void hmm_trainer_model::print() const {
    printf("num_states = %d\nnum_transitions = %d\n",
           num_states,num_transitions);
    printf("initial_state = %d\nfinal_state = %d\n",
           initial_state,final_state);
    if (created) {
      printf("hmm_trainer_model closed\n");
      printf("Transition vector:\n");
      for (int i=0;i<num_transitions;i++) {
        int clst = transition[i].cls_transition;
        printf("tr %3d -> cls_tr %3d, "
               "from %3d to %3d output \"%s\"\n",
               i,clst,
               transition[i].from,
               transition[i].to,
               ((transition[i].output==0) ? "" : transition[i].output));
        printf("          cls_tr %3d emission %d prob %f\n",
               clst,
               trainer->get_cls_transition_emission(clst),
               trainer->get_cls_transition_prob(clst).to_float());
      }
    } else {
      printf("hmm_trainer_model to be closed\n");
      printf("List of transitions before closing the model:\n");
      for (const hmm_aux_transition *r = list_transitions;
           r != 0; r = r->next)
        printf("from %3d to %3d with emission %3d and"
               " id %3d, output=\"%s\"\n",
               r->from,r->to,r->emission,r->id,
               ((r->output==0) ? "" : r->output));
    }
  }

  void hmm_trainer_model::print_dot() const {
    if (created) {
      printf("digraph automata {\n");
      for (int i=0;i<num_states;i++)
        printf(" %d [label=\"%d\"];\n",i,i);
      for (int i=0;i<num_transitions;i++) {
        int clst = transition[i].cls_transition;
        printf(" %3d -> %3d [label=\"<%d> %d/%f/%s\"];\n",
               transition[i].from,
               transition[i].to,
               clst,
               trainer->get_cls_transition_emission(clst),
               trainer->get_cls_transition_prob(clst).to_float(),      
               ((transition[i].output==0) ? "" : transition[i].output));
      }
      printf("}\n");
    }
  }

  // clase auxiliar para imprimir la salida
  struct list_output {
    char *output;
    list_output *next;
  };

  log_float hmm_trainer_model::viterbi(const MatrixFloat *emission,
                                       bool emission_in_log_base,
                                       bool do_expectation,
                                       MatrixFloat *reest_emission,
                                       MatrixFloat *seq_reest_emission,
                                       MatrixFloat *state_probabilities,
                                       char **output_str,
                                       float count_value) {
    log_float logf_count_value  = log_float::from_float(count_value);
    log_double logd_count_value = log_double::from_double((double)count_value);
    //
    int length_sequence  = emission->getDimSize(0);
    int sz_emission_frame= emission->getDimSize(1);

    int sizepath         = (length_sequence+1)*num_states;
    int *path            = new int[sizepath];
    for (int i=0; i<sizepath; i++) path[i] = -1;

    log_float *probnow   = new log_float[num_states];
    log_float *probnxt   = new log_float[num_states];
    log_float *vemission = new log_float[sz_emission_frame];
    const log_float *apriori = trainer->apriori_cls_emission;
    float *femission; // una fila matriz emission
    int *fpath; // recorre fila sq matriz path

    int sq; // recorre secuencia
    int tr; // recorre transiciones
    int st; // recorre estados

    // inicializar probnow a las probabilidades etapa actual, pero lo
    // ponemos en la siguiente pq se intercambian los vectores:
    for (st=0; st<num_states; st++)
      probnxt[st] = log_float::zero();
    probnxt[initial_state] = log_float::one();
  
    // iterator for matrix traversal (each row is a emission frame)
    MatrixFloat::const_iterator emiss_it(emission->begin());
    // bucle ppal:
    for (sq=0,fpath=path+num_states; sq<length_sequence; sq++,fpath+=num_states) {
    
      // preparar vector vemission:
      if (!emission_in_log_base) {
        for (int i=0; i<sz_emission_frame; i++, ++emiss_it)
          vemission[i] = log_float::from_float(*emiss_it) / apriori[i];
      } else {
        for (int i=0; i<sz_emission_frame; i++, ++emiss_it)
          vemission[i] = log_float(*emiss_it) / apriori[i];
      }

      // intercambiar vectores probnow <--> probnxt
      log_float *swap = probnow; probnow = probnxt; probnxt = swap;

      // inicializar vectores probnxt
      for (st=0; st<num_states; st++)
        probnxt[st] = log_float::zero();

      // recorrer transiciones
      for (tr=0; tr<num_transitions; tr++) {
        int orig = transition[tr].from;
        int dest = transition[tr].to;
        int emis = transition_emission(tr);

        log_float nscr = probnow[orig] * transition_prob(tr);

        if (emis >= 0) { // transicion no lambda
          nscr *= vemission[emis];
          if (nscr > probnxt[dest]) { // maximizar prob
            probnxt[dest] = nscr;
            fpath[dest]   = tr;
          }
        } else { // transicion lambda
          if (nscr > probnow[dest]) { // maximizar prob
            probnow[dest] = nscr;
            // restamos num_states pq es fila anterior
            fpath[dest-num_states] = tr;
          }
        }
      } // end for tr recorre transiciones
    } // end for sq recorre secuencia

    // transiciones lambda ultima iteracion:
    for (tr=0; tr<num_transitions; tr++) {
      int emis = transition_emission(tr);
      if (emis < 0) { // transicion lambda
        int orig = transition[tr].from;
        int dest = transition[tr].to;
        log_float nscr = probnxt[orig] * transition_prob(tr);
        if (nscr > probnxt[dest]) { // maximizar prob
          probnxt[dest] = nscr;
          // restamos num_states pq es fila anterior
          fpath[dest-num_states] = tr;
        }
      }
    } // end for tr recorre transiciones

    // para recuperar la cadena de salida:
    int outputsz = 0; // longitud de la salida
    int theoutputlistsize = 0;
    list_output *theoutputlist = 0;

    // devolver las probabilidades de cada estado al finalizar
    if (state_probabilities) {
      april_assert(state_probabilities->getNumDim() == 1 &&
                   state_probabilities->getDimSize(0) == num_states);
      MatrixFloat::iterator st_prob_it(state_probabilities->begin());
      for (int i=0;i<num_states;i++, ++st_prob_it)
        *st_prob_it = probnxt[i];
    }

    if (reest_emission) {
      AprilMath::MatrixExt::Initializers::matZeros(reest_emission);
    }
  
    // recuperar prob final:
    //printf("recuperar prob final\n");
    log_float output_prob = probnxt[final_state];
    // recuperar el camino desde final_state usando la matriz path
    sq = length_sequence-1; fpath=path+length_sequence*num_states;
    st = final_state;
    tr = fpath[st];
    while (tr >= 0) {

      if (do_expectation) {
        // acumular valores para algoritmo EM
        int clstr = transition[tr].cls_transition;
        trainer->acum_tran_prob(clstr,
                                logf_count_value);
      }

      int emis  = transition_emission(tr);
      if (emis >= 0) { 
        // acumular para calcular prob. a priori de las emisiones
        if (do_expectation) trainer->acum_cls_emission[emis] += logd_count_value;
        // guardar la emision:
        if (seq_reest_emission) (*seq_reest_emission)(sq)  = emis+1;
        if (reest_emission)     (*reest_emission)(sq,emis) = 1.0f;
      }

      // salida:
      if (transition[tr].output != 0) {
        list_output *aux = new list_output;
        aux->output = transition[tr].output;
        aux->next = theoutputlist;
        theoutputlist = aux;
        theoutputlistsize++;
        outputsz += strlen(transition[tr].output);
      }
    
      // pasar al estado anterior:
      if (emis >= 0) {
        sq--; fpath-=num_states;
      }
      st = transition[tr].from;
      tr = fpath[st];
    } // while (tr >= 0);

    // generar cadena de salida:
    if (theoutputlistsize > 0)
      outputsz += theoutputlistsize-1;
    outputsz++; // para el '\0'

    char *outputstr = new char[outputsz];
    char *r = outputstr;
    while (theoutputlist) {
      list_output *aux = theoutputlist;
      theoutputlist = theoutputlist->next;
      strcpy(r,aux->output);
      r += strlen(aux->output);
      if (theoutputlist) {
        *r = ' '; r++;
      }
      delete aux;
    }
    *r = '\0';
    *output_str = outputstr;

    // liberar recursos
    delete[] path;
    delete[] probnow;
    delete[] probnxt;
    delete[] vemission;
  
    // devolver maxprob
    return output_prob;

  } // end viterbi method

  void hmm_trainer_model::forward(MatrixFloat *emission,
                                  log_float *alpha) {

    // alpha es una matriz de tamanyo: length_sequence+1 filas por
    // num_states columnas creada en el metodo forward_backward donde
    // guardamos probabilidades denominadas usualmente "alpha" en la
    // literatura, se guardan en formato log_float. Asumimos que esta
    // matriz ha sido inicializada a valor log_float::zero()

    int length_sequence  = emission->getDimSize(0);
    int sz_emission_frame= emission->getDimSize(1);
    log_float *vemission = new log_float[sz_emission_frame];
    const log_float *apriori = trainer->apriori_cls_emission;
    float *femission; // recorre fila matriz emission

    int sq; // recorre secuencia
    int tr; // recorre transiciones

    // probnow recorre desde la fila 0 de la matriz alpha hasta la fila
    // length_sequence-1 que es la penúltima.

    // probnxt recorre desde la fila 1 (la segunda) de la matriz alpha
    // hasta la ultima fila que es length_sequence

    log_float *probnow = alpha;
    log_float *probnxt = 0; // fila siguiente: probnow + num_states;

    // alpha esta inicializada a log_float::zero(), falta este caso
    // inicial:
    probnow[initial_state] = log_float::one();

    // bucle ppal
    for (sq=0, femission=emission->getRawDataAccess()->getPPALForReadAndWrite();
         sq<length_sequence;
         sq++, femission+=sz_emission_frame, probnow+=num_states) {

      // preparar vector vemission:
      for (int i=0; i<sz_emission_frame; i++)
        vemission[i] = log_float::from_float(femission[i]) / apriori[i];

      // probnxt es la fila siguiente:
      probnxt = probnow + num_states;

      // recorrer transiciones
      for (tr=0; tr<num_transitions; tr++) {
        int orig = transition[tr].from;
        int dest = transition[tr].to;
        int emis = transition_emission(tr);
        log_float nscr = probnow[orig] * transition_prob(tr);
        if (emis >= 0) // transicion no lambda
          probnxt[dest] += nscr*vemission[emis];
        else           // transicion lambda
          probnow[dest] += nscr;
      }

    } // end for sq recorre secuencia

    // caso especial: tratar las transiciones lambda de la ultima
    // iteracion, ultima fila de la matriz (apuntada por probnxt)

    for (tr=0; tr<num_transitions; tr++) {
      int orig = transition[tr].from;
      int dest = transition[tr].to;
      int emis = transition_emission(tr);
      if (emis < 0) // si es transicion lambda
        probnxt[dest] += probnxt[orig] * transition_prob(tr);
    }

    delete[] vemission;

  } // end forward method

  void hmm_trainer_model::backward(MatrixFloat *input_emission, 
                                   MatrixFloat *output_emission, 
                                   log_float *alpha,
                                   bool do_expectation) {

    // alpha es una matriz de tamanyo: length_sequence+1 filas por
    // num_states columnas creada en el metodo forward_backward donde se
    // supone que el método forward ya ha guardado en ella los valores
    // alpha en formato log_float:

    // la fila 0 de alpha contiene la probabilidad alpha antes de haber
    // consumido ninguna trama, ... la fila i contiene el valor alpha
    // tras haber consumido i-1 tramas, por tanto la última trama no ha
    // sido utilizada en forward

    int length_sequence     = input_emission->getDimSize(0);
    int sz_emission_frame   = input_emission->getDimSize(1);
    const log_float *apriori= trainer->apriori_cls_emission;

    // femission recorre filas de la matriz emission desde la ultima
    // hasta la primera:
    float *femission=
      input_emission->getRawDataAccess()->getPPALForReadAndWrite()+(length_sequence-1)*sz_emission_frame;
    float *desired_emission =
      output_emission->getRawDataAccess()->getPPALForReadAndWrite()+(length_sequence-1)*sz_emission_frame;

    // recorre la matriz alpha desde la PENULTIMA fila, que corresponde
    // a haber consumido todas las tramas menos una:
    log_float *f_alpha = alpha+(length_sequence-1)*num_states;

    int sq; // recorre secuencia
    int tr; // recorre transiciones
    int st; // recorre estados

    // a diferencia del metodo forward, probnow y probprv son dos
    // vectores que vamos intercambiando, se llama probprv porque vamos
    // hacia atrás:
    log_float *probnow  = new log_float[num_states];
    log_float *probprv  = new log_float[num_states];
    // para calcular la salida deseada, que luego se copiara a emission:
    log_float *desired  = new log_float[sz_emission_frame];
    // para pasar emission a base logaritmica y dividir por a priories:
    log_float *vemission= new log_float[sz_emission_frame];

    // inicializar probprv (que luego sera probnow por el swap)
    // en Jelinek paso 2 pagina 32 dice inicializar todo a uno :|
    for (st=0; st<num_states; st++)
      probprv[st] = log_float::zero();
    probprv[final_state] = log_float::one(); // ojito, es final_state

    for (sq=0; sq<length_sequence; sq++) { // bucle ppal length_sequence
      // veces
    
      // preparar vector vemission:
      for (int i=0; i<sz_emission_frame; i++)
        vemission[i] = log_float::from_float(femission[i]) / apriori[i];

      // intercambiar vectores probprv <--> probnow
      log_float *swap = probnow; probnow = probprv; probprv = swap;

      // inicializar vector probprv
      for (st=0; st<num_states; st++)
        probprv[st] = log_float::zero();

      // recorrer transiciones AL REVES por lo del orden topologico para
      // tratar bien las transiciones lambda:
      for (tr=num_transitions-1; tr>=0; tr--) {
        int orig = transition[tr].to;   // puesto deliberadamente al reves
        int dest = transition[tr].from; // puesto deliberadamente al reves
        int emis = transition_emission(tr);

        log_float nscr = probnow[orig] * transition_prob(tr);
        if (emis >= 0) // transicion no lambda
          probprv[dest] += nscr*vemission[emis];
        else           // transicion lambda
          probnow[dest] += nscr;

      } // end for tr recorre transiciones

      // ahora tenemos en provprv el de la iteracion siguiente a falta
      // de las transiciones lambda, y de regalo tenemos el de la actual
      // YA con sus transiciones lambda:

      // inicializar vector desired:
      for (int i=0; i<sz_emission_frame; i++)
        desired[i] = log_float::zero();

      // utilizar f_alpha y probprv para actualizar la información en la
      // matriz de emision utilizando P^*{t^i=t} (capitulo 2 Jelinek 1997)
      if (do_expectation)
        for (tr=0; tr<num_transitions; tr++) {
          int orig = transition[tr].from;
          int dest = transition[tr].to;
          int emis = transition_emission(tr);
          int clstr= transition[tr].cls_transition;
          log_float pstar;
          if (emis >= 0) { // no es transicion lambda
            pstar = f_alpha[orig] * transition_prob(tr) * 
              probnow[dest] * vemission[emis];
            desired[emis] += pstar; // salida deseada para entrenar posterioris
            trainer->acum_tran_prob(clstr, pstar); // prob. transicion
          } else { // transicion lambda utiliza alpha de la fila siguiente:
            pstar = f_alpha[num_states+orig] * transition_prob(tr) * probnow[dest];
            trainer->acum_tran_prob(clstr, pstar); // prob. transicion
          }
        } // for q recorre transiciones
      else // solamente el calculo de la salida deseada
        for (tr=0; tr<num_transitions; tr++) {
          int orig = transition[tr].from;
          int dest = transition[tr].to;
          int emis = transition_emission(tr);
          log_float pstar;
          if (emis >= 0) { // no es transicion lambda
            pstar = f_alpha[orig] * transition_prob(tr) * 
              probnow[dest] * vemission[emis];
            desired[emis] += pstar; // salida deseada para entrenar posterioris
          }
        } // for q recorre transiciones

      // ya podemos reescribir la fila con la salida deseada:
      log_float acum= desired[0];
      for (int i=1; i<sz_emission_frame; i++)
        acum += desired[i];
      for (int i=0; i<sz_emission_frame; i++) {
        log_float aux = desired[i] / acum;
        trainer->acum_apriori_cls_emission(i,aux);
        desired_emission[i] = aux.to_float();
      }

      // pasamos a la fila de emision anterior:
      femission        -= sz_emission_frame;
      desired_emission -= sz_emission_frame;
      f_alpha          -= num_states;
    
    } // end for sq que cuenta la secuencia

    if (do_expectation) {
      // caso especial que se hace aparte: calculamos beta en la columna
      // inicial del trellis, correspondiente a no haber consumido ninguna
      // trama, que se utiliza para estimar las probabilidades de
      // transitar en las transiciones lambda en el instante inicial:
    
      // recorrer transiciones AL REVES por lo del orden topologico para
      // tratar bien las transiciones lambda:
      for (tr=num_transitions-1; tr>=0; tr--) {
        int orig = transition[tr].to;   // puesto deliberadamente al reves
        int dest = transition[tr].from; // puesto deliberadamente al reves
        int emis = transition_emission(tr);
        if (emis < 0) // transicion lambda
          probprv[dest] += probprv[orig] * transition_prob(tr);
      } // end for tr recorre transiciones
    
      // utilizar f_alpha y probprv para actualizar la información en la
      // matriz de emision utilizando P^*{t^i=t} (capitulo 2 Jelinek 1997)
      f_alpha   += num_states; // deshacer
      for (tr=0; tr<num_transitions; tr++) {
        int orig = transition[tr].from;
        int dest = transition[tr].to;
        int emis = transition_emission(tr);
        int clstr= transition[tr].cls_transition;
        log_float pstar;
        if (emis < 0) { // transicion lambda
          pstar = f_alpha[orig] * transition_prob(tr) * probprv[dest];
          trainer->acum_tran_prob(clstr, pstar); // prob. transicion
        }
      } // for q recorre transiciones
    }

    delete[] probnow;
    delete[] probprv;
    delete[] vemission;
    delete[] desired;
  
  } // end method

  void hmm_trainer_model::forward_backward(MatrixFloat *input_emission, 
                                           MatrixFloat *output_emission, 
                                           bool do_expectation) {
  
    int length_sequence  = input_emission->getDimSize(0);
    // todo: comprobar que output_emission tiene las mismas dim.
    int alpha_size       = (length_sequence+1)*num_states;
    log_float *alpha     = new log_float[alpha_size];
    for (int i=0; i<alpha_size; i++)
      alpha[i] = log_float::zero();
  
    forward(input_emission,alpha);
    backward(input_emission,output_emission,alpha,do_expectation);
  
    delete[] alpha;
  }

} // namespace HMMs
