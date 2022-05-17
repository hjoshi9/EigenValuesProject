#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
//#include <mpi.h>
#include <sys/time.h>


void printExecutionTime(double start, double end){
	printf("Execution time : %lf\n",end-start);
	//exit(1);
}

void getWallTime(double *wc){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	*wc = (double) (tp.tv_sec + tp.tv_usec/1000000.0);
}

double calcMFlops(double start, double end, long fop){
	double timeTaken = end-start;
	if(timeTaken == 0) return 0;
	else return (double)fop/(timeTaken*1e6);
}

// Returns a randomized vector containing N elements
double *get_random_vector(int N) {
	
    // Allocate memory for the vector
    double *V = (double *) malloc(N * sizeof(double));
	
    // Populate the vector with random numbers
    for (int i = 0; i < N; i++) 
      V[i] = rand() % 1000 + 1;
  	// Normalize V
  	double V_norm = 0.0;
	double sum = 0.0;
	for(int i = 0; i<N; i++)
		sum +=V[i]*V[i];
	V_norm = sqrt(sum);
	for(int i = 0; i<N; i++)
		V[i] = V[i]/V_norm;
    //V[i] = (int) rand() / (int) rand();
	
    // Return the randomized vector
    return V;
}

double **generateMatrix(int row, int col){
    srand(time(NULL));

    double **matrix = malloc(row * sizeof(double*));
    for(int i=0;i<row;i++){
    	matrix[i] = malloc(col * sizeof(double));
    }

  	for (int   i = 0; i < row;  i++) {
    	for (int j = 0; j < i + 1; j++) {
      		int num = rand() % 100 + 1;
      		matrix[i][j] = num;
      		matrix[j][i] = num;
    }
  }
  return matrix;
}

void print_array(double **A,int row, int col){
	//print array
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			printf("%lf ",A[i][j] );
		}
		printf("\n");
	}
}

void print_vector(double *A,int row){
	//print array
	for(int i=0;i<row;i++)
			printf("%lf ",A[i] );
		printf("\n");
}

void check(double **A, double *x, int size, double lambda){
	for(int i=0; i<size;i++){
			double sum = 0.0;
			for(int j=0;j<size;j++){
				sum += A[i][j]*x[j];
			}
			if(sum != lambda*x[i]){
				printf("Wrong Eigen value!!");
				break;
			}
		}
}

double power_iteration_serial(double **A, double *x, int size,double tol){
	printf("For Power iteration serial\n");
	int maxiter = 100; // Initialize max error
	double lambda_old = 0.0;
	double lambda_new = 0.0;
	double *x_new = (double *) calloc(size, sizeof(double));
	for(int iter=0; iter<maxiter; iter++){
		//printf("%d, %lf\n",iter,lambda_new);
		// Calculate x' = Ax
		for(int i=0; i<size;i++){
			double sum = 0.0;
			for(int j=0;j<size;j++){
				sum += A[i][j]*x[j];
			}
			x_new[i] =sum;
		}
		// Find Rayleigh quotient (eigen value)
		lambda_new = 0.0;
		for(int i = 0; i<size; i++)
			lambda_new += x_new[i]*x[i];
		
		// Check against tolerance
		if(fabs(lambda_new-lambda_old)<tol){
			break;
		}
		else{
			lambda_old=lambda_new;	
			//Normalize x_new
			double x_new_norm = 0.0;
			double sum = 0.0;
			for(int i = 0; i<size; i++)
				sum +=x_new[i]*x_new[i];
			x_new_norm = sqrt(sum);
			for(int i = 0; i<size; i++)
				x[i] = x_new[i]/x_new_norm;		
			}
	}

	return lambda_new;
}

double power_iteration_omp(double **A, double *x, int size,double tol){
	printf("For Power iteration OPENMP\n");
	int maxiter = 100; // Initialize max error
	double lambda_old = 0.0;
	double lambda_new = 0.0;
	double *x_new = (double *) calloc(size, sizeof(double));
	for(int iter=0; iter<maxiter; iter++){
		//printf("%d, %lf\n",iter,lambda_new);
		// Calculate x' = Ax
		#pragma omp parallel for
		for(int i=0; i<size;i++){
			double sum = 0.0;
			#pragma omp parallel for reduction(+:sum)
			for(int j=0;j<size;j++){
				sum += A[i][j]*x[j];
			}
			x_new[i] =sum;
		}
		// Find Rayleigh quotient (eigen value)
		lambda_new = 0.0;
		#pragma omp parallel for reduction(+:lambda_new)
		for(int i = 0; i<size; i++)
			lambda_new += x_new[i]*x[i];
		
		// Check against tolerance
		if(fabs(lambda_new-lambda_old)<tol){
			break;
		}
		else{
			lambda_old=lambda_new;	
			//Normalize x_new
			double x_new_norm = 0.0;
			double sum = 0.0;
			//#pragma omp parallel for reduction(+:sum)
			for(int i = 0; i<size; i++)
				sum +=x_new[i]*x_new[i];
			x_new_norm = sqrt(sum);
			//#pragma omp parallel for
			for(int i = 0; i<size; i++)
				x[i] = x_new[i]/x_new_norm;		
			}
	}

	return lambda_new;
}



void jacobi_serial(double **A, int size,double tol){
	printf("Jacobi method serial\n");
	int maxiter = 200;
	for(int iter=0; iter<maxiter;iter++){
		double max = 0.0;
		int a = 0;
		int b = 0;
		// Find max element
		for(int i=1;i<size;i++){
			for(int j=0;j<=i-1;j++){
				if(fabs(A[i][j])>max){
					max = fabs(A[i][j]);
					a = i;
					b = j;
				}
			}
		}
		// Rotation angle
		double theta = 0.0;
		if(A[a][a]==A[b][b])
			theta = 3.14159265358979323846/4;
		else
			theta = 0.5*atan(2*A[a][b]/(A[b][b]-A[a][a]));
		// Givens rotation matrix
		double **Q = malloc(size * sizeof(double*));
	    for(int i=0;i<size;i++){
	    	Q[i] = malloc(size * sizeof(double));
	    }
		for(int i=0;i<size;i++){
			for(int j=0;j<size;j++){
				if((i==j)&&(i!=a)&&(i!=b))
					Q[i][j] = 1.0;
				else if((i==j)&&((i==a)||(i==b)))
					Q[i][j] = cos(theta); 
				else
					Q[i][j] = 0.0;
			}
		}
		Q[a][b] = sin(theta);
		Q[b][a] = -sin(theta);
		// Diagonalize A
		double **D = malloc(size * sizeof(double*));
	    for(int i=0;i<size;i++){
	    	D[i] = malloc(size * sizeof(double));
	    }
		for(int i=0;i<size;i++){
			for(int j=0;j<size;j++){
				double sum = 0.0;
				for(int k=0;k<size;k++){
					for(int l=0;l<size;l++){
						sum += Q[k][i]*Q[l][j]*A[k][l];
					}
				}			
				D[i][j] = sum;
			}
		}
		// A = D
		for(int i=0;i<size;i++){
			for(int j=0;j<size;j++){
				A[i][j] = D[i][j];
			}
		}
		double err = 0.0;
		for(int i=0;i<size;i++){
			for(int j=0;j<size;j++){
				if(i!=j)
					err += A[i][j]*A[i][j];
			}
		}
		if(sqrt(err)<tol){
			free(Q);
			free(D);
			break;
		}
	}

}

void jacobi_omp(double **A, int size,double tol){
	printf("Jacobi omp method\n");
	int maxiter = 200;
	for(int iter=0; iter<maxiter;iter++){
		double max = 0.0;
		int a = 0;
		int b = 0;
		// Find max element
		#pragma omp parallel for 
		for(int i=1;i<size;i++){
			//#pragma omp parallel for reduction(max:max)
			for(int j=0;j<=i-1;j++){
				if(fabs(A[i][j])>max){
					max = fabs(A[i][j]);
					a = i;
					b = j;
				}
			}
		}
		// Rotation angle
		double theta = 0.0;
		if(A[a][a]==A[b][b])
			theta = 3.14159265358979323846/4;
		else
			theta = 0.5*atan(2*A[a][b]/(A[b][b]-A[a][a]));
		// Givens rotation matrix
		double **Q = malloc(size * sizeof(double*));
	  for(int i=0;i<size;i++){
	    	Q[i] = malloc(size * sizeof(double));
	  }
	  #pragma omp parallel for collapse(2)
		for(int i=0;i<size;i++){
			for(int j=0;j<size;j++){
				if((i==j)&&(i!=a)&&(i!=b))
					Q[i][j] = 1.0;
				else if((i==j)&&((i==a)||(i==b)))
					Q[i][j] = cos(theta); 
				else
					Q[i][j] = 0.0;
			}
		}
		Q[a][b] = sin(theta);
		Q[b][a] = -sin(theta);
		// Diagonalize A
		double **D = malloc(size * sizeof(double*));
	    for(int i=0;i<size;i++){
	    	D[i] = malloc(size * sizeof(double));
	    }
	  #pragma omp parallel for collapse(2)
		for(int i=0;i<size;i++){
			for(int j=0;j<size;j++){
				double sum = 0.0;
				//#pragma omp parallel for collapse(2) reduction(+:sum)
				for(int k=0;k<size;k++){
					for(int l=0;l<size;l++){
						sum += Q[k][i]*Q[l][j]*A[k][l];
					}
				}			
				D[i][j] = sum;
			}
		}
		// A = D
		#pragma omp parallel for collapse(2)
		for(int i=0;i<size;i++){
			for(int j=0;j<size;j++){
				A[i][j] = D[i][j];
			}
		}
		double err = 0.0;
		#pragma omp parallel for collapse(2) reduction(+:err)
		for(int i=0;i<size;i++){
			for(int j=0;j<size;j++){
				if(i!=j)
					err += A[i][j]*A[i][j];
			}
		}
		if(sqrt(err)<tol){
			free(Q);
			free(D);
			break;
		}
	}

}


int main(int argc, char *argv[]) {

	int row, col;
	row = col = atoi(argv[1]);
	int algo = atoi(argv[2]);
	int num_threads = atoi(argv[3]);
	if(argc!=4){
		printf("Wrong input parameter, need to pass the number of rows for a square array and the algorithm to use");
		return 0;
	}
	omp_set_num_threads(num_threads);
	printf("Size of matrix = %d x %d\n",row,col);
	double **A = generateMatrix(row,col);
	//print_array(A,row,col);
	double tol = 1e-10;
	if(algo==1){
		printf("\nPower iteration \n");
		double *x = get_random_vector(row);
		double start, end = 0.0;
		getWallTime(&start);
		double lambda = power_iteration_serial(A,x,row,tol);
		getWallTime(&end);
		printExecutionTime(start, end);
		start, end = 0.0;
		getWallTime(&start);
		printf("YES!!!!!!\n");
		double lambda_omp = power_iteration_omp(A,x,row,tol);
		getWallTime(&end);
		printExecutionTime(start, end);
		free(A);
		free(x);	
	}
	else{
		printf("\n\n Jacobi Method:\n\n");
		double start, end = 0.0;
		getWallTime(&start);
		jacobi_serial(A,row,tol);
		getWallTime(&end);
		printExecutionTime(start, end);
		print_array(A,row,row);
		double **B = generateMatrix(row,col);
		start, end = 0.0;
		getWallTime(&start);
		jacobi_omp(B,row,tol);
		getWallTime(&end);
		printExecutionTime(start, end);
		print_array(A,row,row);
		free(A);
		free(B);
	}
	
	//return 0;
}
