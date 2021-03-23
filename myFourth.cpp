//#include "ap_axi_sdata.h" // ap_axis can also be used, but it will include all sideband signals which we don't need
#include "hls_stream.h"
#include "ap_int.h"
#include "hls_math.h"

#define INPUT_SIZE 384
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define OUTPUT_SIZE 10

struct AXIS_wLAST{
	float data;
	bool last;
};

float relu(float x) {
	if (x >= 0) {
		return x;
	}
	return 0;
}

#define N 8
void top(int a[N], int b[N], int &out)
{
  int a_int[N],b_int[N];
#pragma HLS array_partition  variable=a_int dim=1 complete
#pragma HLS array_partition  variable=b_int dim=1 complete
  int product = 0;

  for(int i=0;i<N;i++) {
  #pragma HLS pipeline
    a_int[i] = a[i];
  }
  for(int i=0;i<N;i++) {
  #pragma HLS pipeline
    b_int[i] = b[i];
  }

  for(int i=0;i<N;i++) {
  #pragma HLS unroll
    product += a_int[i] * b_int[i];
  }

 out = product;

}

int dot_product(int a[], int b[], int n) {
	int sum = 0;
	for (int l = 0; l < n; l++) {
		sum += a[l] * b[l] / 255;
	}
	return sum;
}

void softmax(int a[]) {
	return;
}

void myip_MLP_quant_HLS(hls::stream<AXIS_wLAST>& S_AXIS, hls::stream<AXIS_wLAST>& M_AXIS){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=S_AXIS
#pragma HLS INTERFACE axis port=M_AXIS

	int inputs[INPUT_SIZE] = {};

	int i = 0;
	int j = 0;
	AXIS_wLAST read_input = {};
	AXIS_wLAST write_output = {};


	// Read layer
	myip_read_hls: for (i = 0; i < INPUT_SIZE; i++) {
		read_input = S_AXIS.read();
		inputs[i] = read_input.data;
	}

	// Hidden layer 1
	myip_hidden1_hls: for (i = 0; i < HIDDEN1_SIZE; i++) {
		hidden1[i] = dot_product(hidden1_matrix[i], inputs, INPUT_SIZE) + hidden1[i];
		hidden1[i] = relu(hidden1[i]);
	}

	//Hidden layer 2
	myip_hidden2_hls: for (i = 0; i < HIDDEN2_SIZE; i++) {
		hidden2[i] = dot_product(hidden2_matrix[i], hidden1, HIDDEN1_SIZE) + hidden2[i];
		hidden2[i] = relu(hidden2[i]);
	}

	//Output layer
	myip_output_hls: for (i = 0; i < OUTPUT_SIZE; i++) {
		outputs[i] = dot_product(output_matrix[i], hidden2, HIDDEN2_SIZE) + outputs[i];
	}
	softmax(outputs);

	//Output to stream
	for (i = 0; i < OUTPUT_SIZE; i++) {
		write_output.data = outputs[i];
		write_output.last = 0;
		if (i == 15) { //build branch predictor here
			write_output.last = 1;
		}
		M_AXIS.write(write_output);
	}
	for (i = OUTPUT_SIZE; i < 16; i++) {
		write_output.data = i;
		if (i == 15) { //build branch predictor here
			write_output.last = 1;
		}
		M_AXIS.write(write_output);
	}
}
