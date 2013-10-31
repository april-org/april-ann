// g++ -g -o gs-pca gs-pca.cc  `pkg-config --libs gsl` -fpermissive `pkg-config --cflags gsl`
//
// C/C++ example for the CBLAS (GNU Scientific Library)
// implementation of PCA-GS algorithm
//
// M. Andrecut (c) 2008
// includes, system
#include <math.h>
#include <time.h>
// includes, GSL & CBLAS
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
// declarations
int gs_pca_gsl(int, int, int,
	       gsl_matrix *, gsl_matrix *,
	       gsl_matrix *);
int print_results(int, int, int,
		  gsl_matrix *, gsl_matrix *,
		  gsl_matrix *, gsl_matrix *);
// main
int main(int argc, char** argv)
{
  // PCA model: X = TP’ + R
  // input: X, MxN matrix (data)
  // input: M = number of rows in X
  // input: N = number of columns in X
  // input: K = number of components (K<=N)
  // output: T, MxK scores matrix
  // output: P, NxK loads matrix
  // output: R, MxN residual matrix
  int M = 1000, m;
  int N = M/2, n;
  int K = 10;
  printf("\nProblem dimensions: MxN=%dx%d, K=%d", M, N, K);
  // initialize srand and clock
  srand(time(NULL));
  clock_t start=clock();
  double htime;
  // initiallize some random test data X
  gsl_matrix *X = gsl_matrix_alloc (M, N);
  for(m=0; m<M; m++)
    {
      for(n=0; n<N; n++)
	{
	  gsl_matrix_set(X, m, n, rand()/(double)RAND_MAX);
	}
    }
  // allocate memory for T, P, R
  gsl_matrix *T = gsl_matrix_alloc (M, K);
  gsl_matrix *P = gsl_matrix_alloc (N, K);
  gsl_matrix *R = gsl_matrix_alloc (M, N);
  htime = ((double)clock()-start)/CLOCKS_PER_SEC;
  printf("\nTime for data allocation: %f\n", htime);
  // call gs_pca_gsl
  start=clock();
  gsl_matrix_memcpy (R, X);
  gs_pca_gsl(M, N, K, T, P, R);
  htime = ((double)clock()-start)/CLOCKS_PER_SEC;
  printf("\n\nTime for GS-PCA host computation: %f\n", htime);
  // the results are in T, P, R
  // print_results(M, N, K, X, T, P, R);
  // memory clean up and shutdown
  gsl_matrix_free(R);
  gsl_matrix_free(P);
  gsl_matrix_free(T);
  gsl_matrix_free(X);
  printf("\nPress ENTER to exit...\n"); getchar();
  return EXIT_SUCCESS;
}
int gs_pca_gsl(int M, int N, int K,
	       gsl_matrix *T, gsl_matrix *P,
	       gsl_matrix *R)
{
  // PCA model: X = TLP’ + R
  // input: X, MxN matrix (data, copied in R)
  // input: M = number of rows in X
  // input: N = number of columns in X
  // input: K = number of components (K<=N)
  // output: T, MxK left eigenvectors
  // output: P, NxK right eigenvectors
  // output: L, Kx1 eigenvalues
  // output: R, MxN residual
  // maximum number of iterations
  int J = 10000;
  // max error
  double er = 1.0e-7;
  double a;
  int n, k, j;
  // mean center the data
  gsl_vector *U = gsl_vector_calloc (M);
  for(n=0; n<N; n++)
    {
      gsl_blas_daxpy(1.0, &gsl_matrix_column(R, n).vector, U);
    }
  for(n=0; n<N; n++)
    {
      gsl_blas_daxpy(-1.0/N, U, &gsl_matrix_column(R, n).vector);
    }
  // allocate memory fo eigenvalues
  gsl_vector *L = gsl_vector_alloc(K);
  // gs_pca
  for(k=0; k<K; k++)
    {
      gsl_blas_dcopy(&gsl_matrix_column(R, k).vector,
		     &gsl_matrix_column(T, k).vector);
      a = 0.0;
      for(j=0; j<J; j++)
	{
	  gsl_blas_dgemv(CblasTrans, 1.0, R,
			 &gsl_matrix_column(T, k).vector,
			 0.0, &gsl_matrix_column(P, k).vector);
	  if(k>0)
	    {
	      gsl_blas_dgemv(CblasTrans, 1.0,
			     &gsl_matrix_submatrix (P, 0, 0, N, k).matrix,
			     &gsl_matrix_column(P, k).vector, 0.0,
			     &gsl_vector_subvector (U, 0, k).vector);
	      gsl_vector *aux = &gsl_vector_subvector(U,0,k).vector;
	      gsl_blas_dgemv(CblasNoTrans, -1.0,
			     &gsl_matrix_submatrix (P, 0, 0, N, k).matrix,
			     &gsl_vector_subvector (U, 0, k).vector, 1.0,
			     &gsl_matrix_column(P, k).vector);
	    }
	  gsl_blas_dscal(1.0/gsl_blas_dnrm2(&gsl_matrix_column(P, k).vector),
			 &gsl_matrix_column(P, k).vector);
	  gsl_blas_dgemv(CblasNoTrans, 1.0, R, &gsl_matrix_column(P, k).vector,
			 0.0, &gsl_matrix_column(T, k).vector);
	  if(k>0)
	    {
	      gsl_blas_dgemv(CblasTrans, 1.0,
			     &gsl_matrix_submatrix (T, 0, 0, M, k).matrix,
			     &gsl_matrix_column(T, k).vector, 0.0,
			     &gsl_vector_subvector (U, 0, k).vector);
	      gsl_blas_dgemv(CblasNoTrans, -1.0,
			     &gsl_matrix_submatrix (T, 0, 0, M, k).matrix,
			     &gsl_vector_subvector (U, 0, k).vector, 1.0,
			     &gsl_matrix_column(T, k).vector);
	    }
	  gsl_vector_set(L, k, gsl_blas_dnrm2(&gsl_matrix_column(T, k).vector));
	  gsl_blas_dscal(1.0/gsl_vector_get(L, k),
			 &gsl_matrix_column(T, k).vector);
	  if(fabs(a - gsl_vector_get(L, k)) < er*gsl_vector_get(L, k)) break;
	  a = gsl_vector_get(L, k);
	}
      gsl_blas_dger (-gsl_vector_get(L, k),
		     &gsl_matrix_column(T, k).vector,
		     &gsl_matrix_column(P, k).vector, R);
    }
  for(k=0; k<K; k++)
    {
      gsl_blas_dscal (gsl_vector_get(L, k),
		      &gsl_matrix_column(T, k).vector);
    }
  for(k=0; k<K; k++) {
    printf("%f\n",gsl_vector_get(L, k));
  }
  // memory clean up
  gsl_vector_free(L);
  gsl_vector_free(U);
  return EXIT_SUCCESS;
}
int print_results(int M, int N, int K,
		  gsl_matrix *X, gsl_matrix *T,
		  gsl_matrix *P, gsl_matrix *R)
{
  int m, n;
  /* If M < 13 print the results on screen */
  // if(M > 12) return EXIT_SUCCESS;
  printf("\nX\n");
  for(m=0; m<M; m++)
    {
      for(n=0; n<N; n++)
	{
	  printf("%+f ", gsl_matrix_get(X, m, n));
	}
      printf("\n");
    }
  printf("\nT\n");
  for(m=0; m<M; m++)
    {
      for(n=0; n<K; n++)
	{
	  printf("%+f ", gsl_matrix_get(T, m, n));
	}
      printf("\n");
    }
  gsl_matrix *F = gsl_matrix_alloc (K, K);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, T, T, 0.0, F);
  printf("\nT’ * T\n");
  for(m=0; m<K; m++)
    {
      for(n=0; n<K; n++)
	{
	  printf("%+f ", gsl_matrix_get(F, m, n));
	}
      printf("\n");
    }
  gsl_matrix_free(F);
  printf("\nP\n");
  for(m=0; m<N; m++)
    {
      for(n=0; n<K; n++)
	{
	  printf("%+f ", gsl_matrix_get(P, m, n));
	}
      printf("\n");
    }
  gsl_matrix *G = gsl_matrix_alloc (K, K);
  gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, P, P, 0.0, G);
  printf("\nP’ * P\n");
  for(m=0; m<K; m++)
    {
      for(n=0; n<K; n++)
	{
	  printf("%+f ", gsl_matrix_get(G, m, n));
	}
      printf("\n");
    }
  gsl_matrix_free(G);
  printf("\nR\n");
  for(m=0; m<M; m++)
    {
      for(n=0; n<N; n++)
	{
	  printf("%+f ", gsl_matrix_get(R, m, n));
	}
      printf("\n");
    }
  return EXIT_SUCCESS;
}
