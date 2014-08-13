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
#ifndef HMM_TRAINER_H
#define HMM_TRAINER_H

#include "referenced.h"
#include "matrixFloat.h"
#include "logbase.h"

namespace HMMs {

  struct hmm_trainer_cls_transition {
    int emission;
    april_utils::log_float prob;
    april_utils::log_double acum; // contador para algoritmo em
    int next; // lista enlazada en un vector, apuntada por un cls_state,
    // poner next=-1 para finalizar, no usamos punteros porque
    // el vector se redimensiona
  };

  struct hmm_trainer_cls_state {
    int cls_tr_list;
    // cada cls_state apunta a la primera hmm_trainer_cls_transition de
    // su lista enlazada, a menos que valga -1 (null) porque no salen
    // transiciones o las que salen son de tipo fixed.
  };

  struct hmm_trainer_transition {
    int cls_transition;
    int from;
    int to;
    char *output;
  };

  struct hmm_aux_transition {
    int from,to,emission,id;  
    april_utils::log_float prob;
    hmm_aux_transition *next;
    char *output;
  };

  class hmm_trainer_model; // forward declaration

  class hmm_trainer : public Referenced {
    friend class hmm_trainer_model;
    int num_cls_states,      vsz_cls_states;
    int num_cls_transitions, vsz_cls_transitions;
    int num_cls_emissions,   vsz_cls_emissions;
    hmm_trainer_cls_transition *cls_transition;
    hmm_trainer_cls_state      *cls_state;
    april_utils::log_float *apriori_cls_emission;
    april_utils::log_double *acum_cls_emission;

    void acum_tran_prob(int clstr, april_utils::log_float prob) {
      cls_transition[clstr].acum += prob;
    }

  public:
    hmm_trainer();
    ~hmm_trainer();

    // para introducir un modelo quizas sea necesario redimensionar
    // algunos vectores:
    void check_cls_state(int st);
    void check_cls_transition(int tr);
    void check_cls_emission(int emis);

    int new_cls_state();
    int new_cls_transition(int c_st);
    int get_num_cls_emissions() const { return num_cls_emissions; }
    int get_num_cls_transitions() const { return num_cls_transitions; }
    void set_apriori_cls_emission(int i, april_utils::log_float a) {
      apriori_cls_emission[i] = a;
    }
    april_utils::log_float get_apriori_cls_emission(int i) const {
      return apriori_cls_emission[i];
    }
    april_utils::log_float get_cls_transition_prob(int i) const {
      return cls_transition[i].prob;
    }
    void acum_apriori_cls_emission(int i, april_utils::log_float prob) {
      acum_cls_emission[i] += prob;
    }
    void set_cls_transition_emission(int i, int emission) {
      cls_transition[i].emission = emission;
    }
    void set_cls_transition_prob(int i, april_utils::log_float prob) {
      cls_transition[i].prob = prob;
    }
    int get_cls_transition_emission(int i) const {
      return cls_transition[i].emission;
    }

    // metodos para algoritmo em, alineamiento forzado viterbi o
    // baum-welch
    void begin_expectation();
    void end_expectation(bool update_trans_prob=true, 
                         bool update_a_priori_emission=true);

    // para leer vector apriori_cls_emission;
    // TODO

    // para debug
    void print() const;

  };

  class hmm_trainer_model : public Referenced {
    // referencia a su trainer:
    hmm_trainer *trainer;

    bool created; // true una vez "cerrado"
    // lista utilizada mientras se introduce el modelo:
    hmm_aux_transition *list_transitions;

    int num_states;
    int num_transitions;

    // puestos inicialmente a -1 mientras no se cierre el modelo ¿?
    int initial_state;
    int final_state;

    // vector de talla num_transitions, creado por prepare_model:
    hmm_trainer_transition *transition;
    // vector de talla num_states, de momento no se usa:
    // int *ranking;
    // vector de talla num_states+1, de momento no se usa:
    // int *first_transition;

    // auxiliares para algoritmos:
    int transition_emission(int tr);
    april_utils::log_float transition_prob(int tr);

    void forward (basics::MatrixFloat *emission, april_utils::log_float *alpha);
    void backward(basics::MatrixFloat *input_emission, 
                  basics::MatrixFloat *output_emission, 
                  april_utils::log_float *alpha,
                  bool do_expectation);

  public:
    hmm_trainer_model(hmm_trainer *trainer);
    ~hmm_trainer_model();

    // para introducir los modelos:
    int new_state();
    void set_initial_state(int st) { initial_state = st; }
    void set_final_state(int st)   { final_state   = st; }
    void set_cls_state(int st, int cls_st);

    void new_transition(int from, int to, 
                        int emission, 
                        int cls_transition,
                        april_utils::log_float prob,
                        const char *output);

    bool prepare_model(); // llamarlo una vez introducido todo

    april_utils::log_float viterbi(const basics::MatrixFloat *emission,
                                   bool emission_in_log_base,
                                   bool do_expectation,
                                   basics::MatrixFloat *reest_emission,
                                   basics::MatrixFloat *seq_reest_emission,
                                   basics::MatrixFloat *state_probabilities,
                                   char **output_str,
                                   float count_value);

    void forward_backward(basics::MatrixFloat *input_emission, 
                          basics::MatrixFloat *output_emission, 
                          bool do_expectation=true);

    void get_information(int &n_states, int &n_transitions) const {
      n_states = num_states; n_transitions = num_transitions;
    }

    // para debug
    void print() const;
    void print_dot() const;

  };

} // namespace HMMs

#endif // HMM_TRAINER_H

