#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>

void read_matrix(int **r_ptr, int** c_ind,float** v, char*fname,int* r_count,int* v_count){	
	FILE * file;
    	if ((file = fopen(fname, "r+")) == NULL)
	{
	    printf("ERROR: file open failed\n");
	    return;
	}
	
	int column_count,row_count,values_count;
	fscanf(file, "%d %d %d\n",&row_count,&column_count,&values_count);
	*r_count = row_count;
	*v_count = values_count;
	int i;
	int *row_ptr =(int*) malloc((row_count+1) * sizeof(int));
	int *col_ind =(int*) malloc(values_count * sizeof(int));
	for(i=0; i<values_count; i++){
		col_ind[i] = -1;
	}
	float *values =(float*) malloc(values_count * sizeof(float));
	int row,column;
	float value;
	while (1) {
		int ret = fscanf(file, "%d %d %f\n",&row,&column,&value);
		column --;
		row --;
		if(ret == 3){
			row_ptr[row]++;
		} else if(ret == EOF) {
		   	break;
		} else {
		    	printf("No match.\n");
		}
	}
    	rewind(file);
    	int index = 0;
    	int val = 0;
	for(i = 0; i<row_count;i++){
		val = row_ptr[i];
		row_ptr[i] = index;
		index += val;
	}
	row_ptr[row_count] = values_count;
	fscanf(file, "%d %d %d\n",&row_count,&column_count,&values_count);
	i = 0;
	while (1) {
		int ret = fscanf(file, "%d %d %f\n",&row,&column,&value);
		column --;
		row --;
		if(ret == 3){
			while(col_ind[i+row_ptr[row]] != -1){ i++;}
			col_ind[i+row_ptr[row]] = column;
			values[i+row_ptr[row]] = value;
			i=0;
		} else if(ret == EOF) {
		   	break;
		} else {
		    	printf("No match.\n");
		}
	}
    	fclose(file);
    	*r_ptr = row_ptr;
    	*c_ind = col_ind;
    	*v = values;
}

__global__ void mat_vector_multiply(const int num_rows,const int *ptr,const int *indices,const float *data,
				const float *x, float* y){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	int row_start, row_end;
	float dot;
	if(row < num_rows){
		dot = 0;
		row_start = ptr[row];
		row_end = ptr[row + 1];
		for(i = row_start; i < row_end; i++){
			dot+= data[i] * x[indices[i]];
		}
	}
	y[row] += dot;
}

int main (int argc, char* argv[]){
	if ( argc != 5){
		printf( "Incorrect usage");
	}
	else{
		int* row_ptr;
		int* col_ind;
		float* values;
		int r_count, v_count, i, k;
		int thread_num = atoi(argv[1]);
		int repetitions = atoi(argv[2]);
		int mode = atoi(argv[3]);
		char* fname = argv[4];
		read_matrix(&row_ptr, &col_ind, &values, fname, &r_count, &v_count);
		float* x =(float*) malloc(r_count* sizeof(float));
		float* y =(float*) calloc(r_count, sizeof(float));
		for(i = 0; i<r_count;i++){
		    	x[i]= 1.0;
		}
		if(mode == 1){
			fprintf(stdout,"Initial Matrix\n");
			for(i = 0; i<r_count;i++){
		    		if(i+1 < r_count){
		    			for(k = row_ptr[i]; k < row_ptr[i+1];k++){
		    				fprintf(stdout,"%d %d %.10f\n",i+1,col_ind[k]+1,values[k]);
		    			}
		    		}	
		    	}
		    	fprintf(stdout,"Initial Vector\n");
		    	for(i = 0; i<r_count;i++){
		    		fprintf(stdout,"%f\n",x[i]);
		    	}
	  	}
		int *d_row_ptr, *d_col_ind;
		float *d_values, *d_x, *d_y;
		cudaMalloc(&d_row_ptr, r_count*sizeof(int));
		cudaMalloc(&d_col_ind, v_count*sizeof(int));
		cudaMalloc(&d_values, v_count*sizeof(int));
		cudaMalloc(&d_x, r_count*sizeof(float));
		cudaMalloc(&d_y, r_count*sizeof(float));
		cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);	  	
	  	// device inputs;
	  	cudaEventRecord(start);
	  	for(k = 0; k<repetitions; k++){
			cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);
		
			// kernel call
			int blocksize = 64;
			int blocknum = ceil(r_count/blocksize); //number of threads fixed and equal to row count
			mat_vector_multiply <<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
		
			cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
			for(i = 0; i<r_count;i++){
				x[i] = y[i];
			    	y[i]= 0.0;
			}
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		int count = 0;
		if(mode == 1){
			fprintf(stdout,"Resulting Vector\n");
		    	for(i = 0; i<r_count;i++){
		    		if(x[i] != 0){
		    		fprintf(stdout,"%.10f\n",x[i]);
		    		count++;
		    		}
		    	}
		    	fprintf(stdout,"count = %d\n", count);
		}
		fprintf(stdout,"time = %f\n", milliseconds);		
	}
}
