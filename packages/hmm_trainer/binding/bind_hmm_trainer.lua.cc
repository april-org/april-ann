//BIND_HEADER_H
#include "utilLua.h"
#include "hmm_trainer.h"
#include "bind_matrix.h"
//BIND_END

//BIND_LUACLASSNAME hmm_trainer hmm_trainer
//BIND_CPP_CLASS hmm_trainer
//BIND_LUACLASSNAME hmm_trainer_model hmm_trainer_model
//BIND_CPP_CLASS hmm_trainer_model

//BIND_CONSTRUCTOR hmm_trainer hmm_trainer
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(hmm_trainer, new hmm_trainer());
}
//BIND_END

//BIND_DESTRUCTOR hmm_trainer
{
}
//BIND_END

//BIND_METHOD hmm_trainer check_cls_emission
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  int emis;
  LUABIND_GET_PARAMETER(1, int, emis);
  obj->check_cls_emission(emis-1); // restamos 1
}
//BIND_END

//BIND_METHOD hmm_trainer new_cls_state
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(number, obj->new_cls_state());
}
//BIND_END

//BIND_METHOD hmm_trainer new_cls_transition
{
  LUABIND_CHECK_ARGN(==,1);
  int cls_state;
  LUABIND_GET_PARAMETER(1, int, cls_state);
  LUABIND_RETURN(number, obj->new_cls_transition(cls_state));
}
//BIND_END

//BIND_METHOD hmm_trainer get_num_cls_emissions
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(number, obj->get_num_cls_emissions());
}
//BIND_END

//BIND_METHOD hmm_trainer set_a_priori_emissions
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  int len;
  float *vec;
  len = table_to_float_vector(L,&vec);
  if (obj->get_num_cls_emissions() == 0) obj->check_cls_emission(len-1);
  else if (len != obj->get_num_cls_emissions()) {
    lua_pushfstring(L,"hmm_trainer set_a_priori_emission "
		    "method: incorrect number of a prioris: "
		    "%d en lugar de %d",
		    len, obj->get_num_cls_emissions());
    lua_error(L);
  }
  for (int i=0; i<len; i++)
    obj->set_apriori_cls_emission(i,log_float::from_float(vec[i]));

  delete[] vec;
}
//BIND_END

//BIND_METHOD hmm_trainer get_a_priori_emissions
{
  LUABIND_CHECK_ARGN(==,0);
  lua_newtable(L);
  int nclsemis = obj->get_num_cls_emissions();
  for (int i=0; i<nclsemis; i++) {
    lua_pushnumber(L,obj->get_apriori_cls_emission(i).to_float());
    lua_rawseti(L,-2,i+1);
  }
  return 1;
}
//BIND_END

//BIND_METHOD hmm_trainer get_transition_probability
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  int index;
  LUABIND_GET_PARAMETER(1, int, index);
  LUABIND_RETURN(number, obj->get_cls_transition_prob(index).to_float());
}
//BIND_END

//BIND_METHOD hmm_trainer get_transition_logprobability
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  int index;
  LUABIND_GET_PARAMETER(1, int, index);
  LUABIND_RETURN(number, obj->get_cls_transition_prob(index).log());
}
//BIND_END

//BIND_METHOD hmm_trainer print
{
  LUABIND_CHECK_ARGN(==,0);
  obj->print();
}
//BIND_END

//BIND_CONSTRUCTOR hmm_trainer_model
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, hmm_trainer);
  hmm_trainer *the_trainer = lua_tohmm_trainer(L,1);
  LUABIND_RETURN(hmm_trainer_model, new hmm_trainer_model(the_trainer));
}
//BIND_END

//BIND_DESTRUCTOR hmm_trainer_model
{
}
//BIND_END

//BIND_METHOD hmm_trainer_model print
{
  LUABIND_CHECK_ARGN(==,0);
  obj->print();
}
//BIND_END

//BIND_METHOD hmm_trainer_model print_dot
{
  LUABIND_CHECK_ARGN(==,0);
  obj->print_dot();
}
//BIND_END

//BIND_METHOD hmm_trainer begin_expectation
{
  LUABIND_CHECK_ARGN(==,0);
  obj->begin_expectation();
}
//BIND_END

//BIND_METHOD hmm_trainer end_expectation
{
  bool update_trans_prob=true;
  bool update_a_priori_emission=true;
  int argn = lua_gettop(L); // number of arguments
  if (argn == 1 && lua_istable(L,1)) {
    lua_pushstring(L, "update_trans_prob");
    lua_gettable(L, 1);
    if (lua_isboolean(L, -1))
      update_trans_prob = (bool)lua_toboolean(L, -1);
    lua_pop(L,1);
    //
    lua_pushstring(L, "update_a_priori_emission");
    lua_gettable(L, 1);
    if (lua_isboolean(L, -1))
      update_a_priori_emission = (bool)lua_toboolean(L, -1);
    lua_pop(L,1);
  }
  //
  obj->end_expectation(update_trans_prob,update_a_priori_emission);
}
//BIND_END

//BIND_METHOD hmm_trainer_model new_state
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(number, obj->new_state());
}
//BIND_END

//BIND_METHOD hmm_trainer_model set_initial_state
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  int st;
  LUABIND_GET_PARAMETER(1, int, st);
  obj->set_initial_state(st);
}
//BIND_END

//BIND_METHOD hmm_trainer_model set_final_state
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  int st;
  LUABIND_GET_PARAMETER(1, int, st);
  obj->set_final_state(st);
}
//BIND_END

//BIND_METHOD hmm_trainer_model new_transition
{
  LUABIND_CHECK_ARGN(>=,5);
  LUABIND_CHECK_ARGN(<=,6);
  LUABIND_CHECK_PARAMETER(1, number);
  LUABIND_CHECK_PARAMETER(2, number);
  LUABIND_CHECK_PARAMETER(3, number);
  LUABIND_CHECK_PARAMETER(4, number);
  LUABIND_CHECK_PARAMETER(5, number);
  int from, to, emission, cls_trns;
  LUABIND_GET_PARAMETER(1, int, from);
  LUABIND_GET_PARAMETER(2, int, to);
  LUABIND_GET_PARAMETER(3, int, emission);
  emission -= 1; // OJO QUE RESTAMOS 1
  LUABIND_GET_PARAMETER(4, int, cls_trns);
  log_float prb= log_float(lua_tonumber(L,5)); // Va en base logarítmica
  const char *output = 0;
  if (lua_isstring(L,6)) {
    output = lua_tostring(L,6);
  }
  obj->new_transition(from,to,emission,cls_trns,prb,output);
}
//BIND_END

//BIND_METHOD hmm_trainer_model prepare_model
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(bool, obj->prepare_model());
}
//BIND_END

//BIND_METHOD hmm_trainer_model get_num_states_transitions
{
  LUABIND_CHECK_ARGN(==,0);
  int states,transitions;
  obj->get_information(states,transitions);
  LUABIND_RETURN(int, states);
  LUABIND_RETURN(int, transitions);
}
//BIND_END

//BIND_METHOD hmm_trainer_model viterbi
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1,
		     "input_emission",
		     "output_emission_seq",
		     "output_emission",
		     "state_probabilities",
		     "do_expectation",
		     "emission_in_log_base",
		     "count_value",
		     0);
  //
  MatrixFloat *input_matemi, *output_matemi_seq, *output_matemi;
  MatrixFloat *state_probabilities;
  bool do_expectation, emission_in_log_base;
  LUABIND_GET_TABLE_PARAMETER(1, input_emission, MatrixFloat, input_matemi);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output_emission_seq,
				       MatrixFloat, output_matemi_seq, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output_emission,
				       MatrixFloat, output_matemi, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, state_probabilities,
				       MatrixFloat, state_probabilities, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, do_expectation, bool,
				       do_expectation, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, emission_in_log_base, bool,
				       emission_in_log_base, false);
  // Modificar el valor a sumar en cada cuenta, de esa forma se pueden
  // ponderar las cuentas de diferentes frases dando mas pesos a unas
  // que a otras
  float count_value;
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, count_value,
				       float, count_value, 1.0f);
  //
  char *output;
  log_float resul = obj->viterbi(input_matemi,
				 emission_in_log_base,
				 do_expectation,
				 output_matemi,
				 output_matemi_seq,
				 state_probabilities,
				 &output,
				 count_value);
  LUABIND_RETURN(float, resul.log()); // devuelve log(probabilidad)
  LUABIND_RETURN(string, output);
  delete[] output;
}
//BIND_END

//BIND_METHOD hmm_trainer_model forward_backward
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  //
  lua_pushstring(L, "input_emission");
  lua_gettable(L,1);
  if (lua_isnil(L,-1)) {
    lua_pushstring(L,"hmm_trainer_model: input_emission not found");
    lua_error(L);
  }
  MatrixFloat* input_matemi = lua_toMatrixFloat(L,-1);
  // por defecto utiliza la misma matriz para dejar el resultado:
  MatrixFloat* output_matemi = input_matemi;
  lua_pop(L,1);
  if (input_matemi->getNumDim() != 2) {
    lua_pushstring(L,"hmm_trainer model forward_backward method: "
		   "emission matrix must have dim 2");
    lua_error(L);
  }
  //
  lua_pushstring(L, "output_emission");
  lua_gettable(L,1);
  if (!lua_isnil(L,-1)) {
    output_matemi = lua_toMatrixFloat(L,-1);
    // TODO: comprobar matriz es ok
  }
  lua_pop(L,1);
  // por defecto hace do_expectation:
  bool do_expectation = true;
  lua_pushstring(L, "do_expectation");
  lua_gettable(L,1);
  if (!lua_isnil(L,-1)) {
    do_expectation = lua_toboolean(L,-1);
  }
  lua_pop(L,1);
  //
  obj->forward_backward(input_matemi,output_matemi,do_expectation);
}
//BIND_END

//BIND_CLASS_METHOD hmm_trainer to_log
{
  LUABIND_CHECK_ARGN(>=,1);
  int argn = lua_gettop(L); // number of arguments
  if (lua_isMatrixFloat(L,1)) {
    MatrixFloat* mat = lua_toMatrixFloat(L,1);
    log_float *aux = new log_float(mat->getDimSize(1));
    if (argn > 1) {
      for (int i=0;i<mat->getDimSize(1);i++) {
	lua_rawgeti(L, 2, i+1);
	float f = lua_tonumber(L,-1);
	lua_pop(L,1);
	aux[i] = log_float::from_float(f); // TODO: revisar este codigo
	// antes ponia: aux[i] = log_float::from_float(aux[i]);
      }
    } else {
      for (int i=0;i<mat->getDimSize(1);i++) {
	aux[i] = log_float::one();
      }
    }
    MatrixFloat::iterator mat_it(mat->begin());
    for (int i=0;i<mat->getDimSize(0);i++) {
      for (int j=0;j<mat->getDimSize(1);j++,++mat_it)
	*mat_it = (log_float::from_float(*mat_it) / aux[j]).log();
    }
  }
  if (lua_isnumber(L,1)) {
    float num = lua_tonumber(L,1);
    lua_pushnumber(L,log_float::from_float(num).log());
    return 1;
  }
}
//BIND_END

//BIND_CLASS_METHOD hmm_trainer from_log
{
  LUABIND_CHECK_ARGN(==,1);
  if (lua_isMatrixFloat(L,1)) {
    MatrixFloat* mat = lua_toMatrixFloat(L,1);
    for (MatrixFloat::iterator mat_it(mat->begin());
	 mat_it!=mat->end(); ++mat_it)
      *mat_it = log_float(*mat_it).to_float();
  }
  if (lua_isnumber(L,1)) {
    float num = lua_tonumber(L,1);
    lua_pushnumber(L,log_float(num).to_float());
    return 1;
  }
}
//BIND_END

//BIND_FUNCTION HMMTrainer.utils.initial_emission_alignment
  {
    LUABIND_CHECK_ARGN(==,2);
    int npat;
    LUABIND_GET_PARAMETER(2, int, npat);
    lua_pop(L,1);
    int *seq_emission;
    int nsts = table_to_int_vector(L,&seq_emission);

    // creamos una matriz de dim 1 con npat valores:
    MatrixFloat* mat = new MatrixFloat(1, &npat);

    // repartimos los npat valores en nsts posiciones
    int cociente = npat/nsts;
    int resto    = npat%nsts;
    int i,r,v;
    float s;
    for (i=0,r=0; i < nsts; i++) {
      v = cociente;
      if (resto>0) { v++; resto--; }
      s = seq_emission[i];
      for (;v>0;v--) { (*mat)(r) = s; r++; }
    }

    delete[] seq_emission;
    LUABIND_RETURN(MatrixFloat, mat);  
  }
//BIND_END
 
