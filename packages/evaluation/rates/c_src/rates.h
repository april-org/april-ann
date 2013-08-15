#ifndef RATES_H
#define RATES_H

struct int_sequence {
  int size;
  int *symbol;
};

struct pairs_int_sequences {
  int_sequence correct;
  int_sequence test;
  pairs_int_sequences *next;
};

struct counter_edition {
  int na,ns,ni,nb;
  counter_edition(int a=0, int s=0, int i=0, int b=0) :
    na(a),ns(s),ni(i),nb(b) {}
  counter_edition& operator += (const counter_edition &otro) {
    na += otro.na; ns += otro.ns; ni += otro.ni; nb += otro.nb;
    return *this;
  }
};

template <typename T>
class simple_matrix {
public:
  T *m;
  int columns,rows;
  simple_matrix(int rows, int columns);
  ~simple_matrix();
  T *operator[](int row);
  void fill(T v);
};
template <typename T>
simple_matrix<T>::simple_matrix(int rows, int columns) {
  this->columns = columns;
  this->rows    = rows;
  m = new T[rows*columns];
}
template <typename T>
simple_matrix<T>::~simple_matrix() {
  delete[] m;
}
template <typename T>
T* simple_matrix<T>::operator[](int row) {
  return m+(row*columns);
}
template <typename T>
void simple_matrix<T>::fill(T v) {
  int s=columns*rows;
  for (int i=0;i<s;i++) m[i] = v;
}
typedef simple_matrix<int> conf_matrix;

class rates {

  static int dict_size(const pairs_int_sequences *data);

  static counter_edition gp_path(double p,
				 int_sequence correct,
				 int_sequence test,
				 conf_matrix &confmat);

  static counter_edition Gp(double p, 
			    const pairs_int_sequences *data);

  static counter_edition Gp_path(double p, 
				 const pairs_int_sequences *data,
				 conf_matrix &confmat);

  static counter_edition gp_normalized(int_sequence correct,
				       int_sequence test);

  static double initialize_lambda(const pairs_int_sequences *data);

  static double Fp(const pairs_int_sequences *data, 
		   counter_edition &resul);

  static double Fp_path(const pairs_int_sequences *data, 
			counter_edition &resul,
			conf_matrix &confmat);

public:

  static counter_edition gp(double p,
			    int_sequence correct,
			    int_sequence test);

  static double rate(const pairs_int_sequences *data,
		     const char *rate_type,
		     counter_edition &counted,
		     double &p,
		     bool with_p=false,
		     conf_matrix **m = 0);
};

#endif // RATES_H
