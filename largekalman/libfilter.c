#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "utils.c"

//1. Params file generator (forwards and backwards)
//2. Filter generator & write to disk
//3. Smoother backwards generator

/*
typedef struct {
	float *values;
	int width;
	int height;
	bool is_const;
} Matrix;

typedef struct {
	Matrix F;
	Matrix H;
	Matrix Q;
	Matrix R;
	FILE *param_file;
	bool initialised;
	float *buffer;
	int buffer_pos;
	int buffer_end;
	int buffer_capacity;
	int param_line_size;
} Params;

Params get_first_params(FILE *param_file, int buffer_capacity) {
	Params params;

	int param_header[6];
	fread(param_header, sizeof(int), 6, param_file);
	const int n_obs = param_header[0];
	const int n_latents = param_header[1];
	Params.F.is_const = param_header[2] != 0;
	Params.H.is_const = param_header[3] != 0;
	Params.Q.is_const = param_header[4] != 0;
	Params.R.is_const = param_header[5] != 0;

	F.width = n_latents;
	F.height = n_latents;
	H.width = n_obs;
	H.height = n_latents;
	Q.width = n_latents;
	Q.height = n_latents;
	R.width = n_obs;
	R.height = n_obs;

	if (F.is_const) fread(F.values, sizeof(float), F.width*F.height, param_file);
	if (H.is_const) fread(H.values, sizeof(float), H.width*H.height, param_file);
	if (Q.is_const) fread(Q.values, sizeof(float), Q.width*Q.height, param_file);
	if (R.is_const) fread(R.values, sizeof(float), R.width*R.height, param_file);

	float buffer[buffer_size*param_line_size];
	int num_floats_read = fread(buffer, sizeof(float), param_line_size * buffer_size, param_file);

	params.param_file = param_file;
	params.initialised = false;
	params.param_line_size = (F_is_const ? 0 : F_size) + (H_is_const ? 0 : H_size) + (Q_is_const ? 0 : Q_size) + (R_is_const ? 0 : R_size);
	return get_next_params(params);
}

Params get_next_params(Params params) {
	if (!params.initialised) {
		params.buffer = malloc(); //buffer_capacity * param_line_size
		params.buffer_pos = 0;
		params.initialised = true;
	}

}

Params (*stream_params_forwards(FILE *param_file, int buffer_size))() {
	int param_header[6];
	fread(param_header, sizeof(int), 6, param_file);
	const int n_obs = param_header[0];
	const int n_latents = param_header[1];
	const bool F_is_const = param_header[2] != 0;
	const bool H_is_const = param_header[3] != 0;
	const bool Q_is_const = param_header[4] != 0;
	const bool R_is_const = param_header[5] != 0;

	int F_size = n_latents*n_latents;
	int H_size = n_obs*n_latents;
	int Q_size = n_latents*n_latents;
	int R_size = n_obs*n_obs;

	float F_const[F_size];
	if (F_is_const) fread(F_const, sizeof(float), F_size, param_file);
	float H_const[H_size];
	if (H_is_const) fread(H_const, sizeof(float), H_size, param_file);
	float Q_const[Q_size];
	if (Q_is_const) fread(Q_const, sizeof(float), Q_size, param_file);
	float R_const[R_size];
	if (R_is_const) fread(R_const, sizeof(float), R_size, param_file);

	int param_line_size = (F_is_const ? 0 : F_size) + (H_is_const ? 0 : H_size) + (Q_is_const ? 0 : Q_size) + (R_is_const ? 0 : R_size);

	Params yield_params() {
		static float buffer[buffer_size*param_line_size];
		static int num_floats_read = fread(buffer, sizeof(float), param_line_size * buffer_size, param_file);

		static int t = 0;
		static bool inner_loop_done = true;

		while (true) {

		}

		do {
			if (inner_loop_done) t = 0;
			for (; t < num_floats_read/param_line_size; t++) {
				inner_loop_done = false;
				float *F = F_is_const ? F_const : &param_buffer[t*param_line_size];
				float *H = H_is_const ? H_const : &param_buffer[t*param_line_size + (F_is_const ? 0 : F_size)];
				float *Q = Q_is_const ? Q_const : &param_buffer[t*param_line_size + (F_is_const ? 0 : F_size) + (H_is_const ? 0 : H_size)];
				float *R = R_is_const ? R_const : &param_buffer[t*param_line_size + (F_is_const ? 0 : F_size) + (H_is_const ? 0 : H_size) + (Q_is_const ? 0 : Q_size)];
			}
			inner_loop_done = true
			num_params_read = fread(params_buffer, sizeof(Params), param_line_size * buffer_size, param_file);
		} while (!feof(param_file));
	}

	return &yield_params;
}
*/

//Kalman filter
void write_forwards(FILE *obs_file, FILE *param_file, FILE *forw_file, int buffer_size) {
	int obs_header[1];
	fread(obs_header, sizeof(int), 1, obs_file);
	int n_obs = obs_header[0];

	int param_header[6];
	fread(param_header, sizeof(int), 6, param_file);
	//assert (n_obs == param_header[0]);
	const int n_latents = param_header[1];
	const bool F_is_const = param_header[2] != 0;
	const bool H_is_const = param_header[3] != 0;
	const bool Q_is_const = param_header[4] != 0;
	const bool R_is_const = param_header[5] != 0;
	printf("%d %d %d %d %d %d\n",n_obs,n_latents,F_is_const,H_is_const,Q_is_const,R_is_const);

	//Handle constant param reading
	int F_size = n_latents*n_latents;
	int H_size = n_obs*n_latents;
	int Q_size = n_latents*n_latents;
	int R_size = n_obs*n_obs;

	float F_const[F_size];
	if (F_is_const) {
		fread(F_const, sizeof(float), F_size, param_file);
	}
	float H_const[H_size];
	if (H_is_const) {
		fread(H_const, sizeof(float), H_size, param_file);
	}
	float Q_const[Q_size];
	if (Q_is_const) {
		fread(Q_const, sizeof(float), Q_size, param_file);
	}
	float R_const[R_size];
	if (R_is_const) {
		fread(R_const, sizeof(float), R_size, param_file);
	}

	int param_line_size = (F_is_const ? 0 : F_size) + (H_is_const ? 0 : H_size) + (Q_is_const ? 0 : Q_size) + (R_is_const ? 0 : R_size);

	float obs_buffer[buffer_size*n_obs];
	float param_buffer[buffer_size*param_line_size];

	float latents_mu[n_latents];
	float latents_cov[n_latents*n_latents];
	bool latents_initialised = false;

	int obs_floats_read = fread(obs_buffer, sizeof(float), n_obs * buffer_size, obs_file);
	fread(param_buffer, sizeof(float), param_line_size * buffer_size, param_file);
	do {
		for (int t = 0; t < obs_floats_read/n_obs; t++) {
			float *obs = &obs_buffer[t*n_obs];

			float *F = F_is_const ? F_const : &param_buffer[t*param_line_size];
			float *H = H_is_const ? H_const : &param_buffer[t*param_line_size + (F_is_const ? 0 : F_size)];
			float *Q = Q_is_const ? Q_const : &param_buffer[t*param_line_size + (F_is_const ? 0 : F_size) + (H_is_const ? 0 : H_size)];
			float *R = R_is_const ? R_const : &param_buffer[t*param_line_size + (F_is_const ? 0 : F_size) + (H_is_const ? 0 : H_size) + (Q_is_const ? 0 : Q_size)];

			printf("obs ");
			for (int i = 0; i < n_obs; i++) {
				printf("%f ",obs[i]);
			}
			printf("\n");

			if (!latents_initialised){
				//x <- H.T@(H@H.T)^-1@obs
				float HHT[n_obs*n_obs];
				matmul_transposed(H,H,HHT,n_obs,n_latents,n_obs);
				solve(HHT, obs, n_obs, 1);
				matmul(H,obs,latents_mu,n_latents,n_obs,1);
				//P <- 0
				memset(latents_cov, 0, n_latents*n_latents*sizeof(float));
				latents_initialised = true;
			} else {
				//x <- F@x
				float latents_mu_old[n_latents];
				memcpy(latents_mu_old, latents_mu, n_latents*sizeof(float));
				matmul(F,latents_mu_old,latents_mu,n_latents,n_latents,1);
				//P <- F@P@F.T+Q
				float FP[n_latents*n_latents];
				matmul(F,latents_cov,FP,n_latents,n_latents,n_latents);
				matmul_transposed(FP,F,latents_cov,n_latents,n_latents,n_latents);
				vector_plusequals(latents_cov,Q,n_latents*n_latents);
				//K <- ((H@P@H.T+R)^-1@H@P).T	#kalman gain
				float KT[n_obs*n_latents];
				matmul_transposed(H,latents_cov,KT,n_obs,n_latents,n_latents);
				float HP[n_obs*n_latents];
				matmul(H,latents_cov,HP,n_obs,n_latents,n_latents);
				float HPHT_R[n_obs*n_obs];
				matmul_transposed(HP,H,HPHT_R,n_obs,n_latents,n_obs);
				vector_plusequals(HPHT_R,R,n_obs*n_obs);
				solve(HPHT_R,KT,n_obs,n_latents);
				//x <- x + K@(obs - H@x)
				float pred[n_obs];
				matmul(H,latents_mu,pred,n_obs,n_latents,1);
				vector_minusequals(obs,pred,n_obs); //modifies obs
				float latents_update[n_latents];
				matmul_transposed(obs,KT,latents_update,1,n_obs,n_latents);
				vector_plusequals(latents_mu,latents_update,n_latents);
				//P <- (I-K@H)@P
				float KHP_T[n_latents*n_latents];
				matmul_transposed(KT,HP,KHP_T, n_latents, n_obs, n_latents);
				vector_minusequals(latents_cov,KHP_T,n_latents*n_latents);//possible catastrophic cancellation
			}

			printf("latents_mu ");
			for (int i = 0; i < n_latents; i++) {
				printf("%f ",latents_mu[i]);
			}
			printf("\n");
			printf("latents_cov\n");
			for (int i = 0; i < n_latents; i++) {
				for (int j = 0; j < n_latents; j++) {
					printf("%f ",latents_cov[i*n_latents+j]);
				}
				printf("\n");
			}
			printf("\n");
			fwrite(latents_mu, sizeof(float), n_latents, forw_file);
			fwrite(latents_cov, sizeof(float), n_latents*n_latents, forw_file);
		}
		obs_floats_read = fread(obs_buffer, sizeof(float), n_obs * buffer_size, obs_file);
		fread(param_buffer, sizeof(float), param_line_size * buffer_size, param_file);
	} while (!(feof(obs_file) || feof(param_file)));
}

//Backwards step
void write_backwards(FILE *param_file, FILE *obs_file, FILE *forw_file, FILE *backw_file, int buffer_size) {
	printf("calling write_backwards\n");
	int param_header[6];
	//fseek(param_file, 0, SEEK_SET);
	fread(param_header, sizeof(int), 6, param_file);

	int n_obs = param_header[0];
	int n_latents = param_header[1];
	int F_is_const = param_header[2] != 0;
	int H_is_const = param_header[3] != 0;
	int Q_is_const = param_header[4] != 0;
	int R_is_const = param_header[5] != 0;

	int F_size = n_latents * n_latents;
	int H_size = n_obs * n_latents;
	int Q_size = n_latents * n_latents;
	int R_size = n_obs * n_obs;

	float F_const[F_size];
	float H_const[H_size];
	float Q_const[Q_size];
	float R_const[R_size];

	if (F_is_const) {
		fread(F_const, sizeof(float), F_size, param_file);
	}
	if (H_is_const) {
		fread(H_const, sizeof(float), H_size, param_file);
	}
	if (Q_is_const) {
		fread(Q_const, sizeof(float), Q_size, param_file);
	}
	if (R_is_const) {
		fread(R_const, sizeof(float), R_size, param_file);
	}

	long param_data_start = ftell(param_file);

	int param_line_size =
		(F_is_const ? 0 : F_size) +
		(H_is_const ? 0 : H_size) +
		(Q_is_const ? 0 : Q_size) +
		(R_is_const ? 0 : R_size);

	int forw_stride = n_latents + n_latents * n_latents;
	printf("hello! n_latents=%d, forw_stride=%d, buffer_size=%d\n",n_latents,forw_stride,buffer_size);

	float forw_buffer[buffer_size * forw_stride];
	float param_buffer[buffer_size * param_line_size + 1];//+1 to prevent zero-sized VLA

	float latents_mu[n_latents];
	float latents_cov[n_latents * n_latents];

	float latents_mu_pred[n_latents];
	float latents_cov_pred[n_latents * n_latents];

	float latents_mu_smoothed[n_latents];
	float latents_cov_smoothed[n_latents * n_latents];

	float latents_mu_smoothed_next[n_latents];
	float latents_cov_smoothed_next[n_latents * n_latents];

	float latents_cov_lag1[n_latents * n_latents];

	printf("hey\n");
	fseek(forw_file, 0, SEEK_END);
	long end_pos = ftell(forw_file);

	fseek(forw_file, end_pos - sizeof(float) * forw_stride, SEEK_SET);
	fread(latents_mu_smoothed_next, sizeof(float), n_latents, forw_file);
	fread(latents_cov_smoothed_next, sizeof(float), n_latents * n_latents, forw_file);

	//Sufficient statistics
	float latents_mu_smoothed_sum[n_latents];
	memset(latents_mu_smoothed_sum,0,n_latents);
	float latents_cov_smoothed_sum[n_latents*n_latents];
	memset(latents_cov_smoothed_sum,0,n_latents*n_latents);
	float latents_cov_lag1_sum[n_latents*n_latents];
	memset(latents_cov_lag1_sum,0,n_latents*n_latents);
	float obs_prod_latents_mu_smoothed_sum[n_obs*n_latents];
	memset(obs_prod_latents_mu_smoothed_sum,0,n_obs*n_latents);
	int num_datapoints = 0;

	printf("just before do loop starts. forw_stride = %d, n_latents = %d\n",forw_stride,n_latents);
	while (true) {
		printf("start of iter!\n");
		long cur_pos = ftell(forw_file);
		printf("cur_pos = %ld\n",cur_pos);
		long floats_left = cur_pos / sizeof(float);
		int steps = floats_left / forw_stride;
		printf("steps = %d\n",steps);
		if (steps > buffer_size) {
			steps = buffer_size;
		}
		if (steps <= 0) {
			break;
		}

		printf("determining block_start_pos\n");
		long block_start_pos = cur_pos - sizeof(float) * forw_stride * steps;
		if (block_start_pos < 0) {
			break;
		}

		printf("Attempting fseek\n");

		fseek(forw_file, block_start_pos, SEEK_SET);
		fread(forw_buffer, sizeof(float), steps * forw_stride, forw_file);
		fseek(forw_file, block_start_pos, SEEK_SET);

		long t0 = (ftell(forw_file) / sizeof(float)) / forw_stride;

		fseek(param_file,
			  param_data_start + t0 * param_line_size * sizeof(float),
			  SEEK_SET);
		fread(param_buffer, sizeof(float), steps * param_line_size, param_file);

		printf("attempting ftell\n");
		printf("ftell(forw_file) = %ld\n", ftell(forw_file));

		for (int b = steps - 1; b >= 0; b--) {
			printf("loop iter starting. b/steps = %d/%d\n",b,steps);
			float *forw_ptr = &forw_buffer[b * forw_stride];

			memcpy(latents_mu, forw_ptr, sizeof(float) * n_latents);
			memcpy(latents_cov, forw_ptr + n_latents,
				   sizeof(float) * n_latents * n_latents);

			float *F = F_is_const ? F_const :
				&param_buffer[b * param_line_size];

			float *Q = Q_is_const ? Q_const :
				&param_buffer[b * param_line_size +
							  (F_is_const ? 0 : F_size) +
							  (H_is_const ? 0 : H_size)];

			matmul(F, latents_mu, latents_mu_pred,
				   n_latents, n_latents, 1);

			float FP[n_latents * n_latents];
			matmul(F, latents_cov, FP,
				   n_latents, n_latents, n_latents);
			matmul_transposed(FP, F, latents_cov_pred,
							  n_latents, n_latents, n_latents);
			vector_plusequals(latents_cov_pred, Q,
							  n_latents * n_latents);

			float G[n_latents * n_latents];
			matmul_transposed(latents_cov, F, G,
							  n_latents, n_latents, n_latents);

			float latents_cov_pred_copy[n_latents * n_latents];
			memcpy(latents_cov_pred_copy, latents_cov_pred,
				   sizeof(float) * n_latents * n_latents);
			solve(latents_cov_pred_copy, G,
				  n_latents, n_latents);

			memcpy(latents_mu_smoothed,
				   latents_mu_smoothed_next,
				   sizeof(float) * n_latents);
			vector_minusequals(latents_mu_smoothed,
							   latents_mu_pred,
							   n_latents);

			float delta_mu[n_latents];
			matmul(G, latents_mu_smoothed,
				   delta_mu, n_latents, n_latents, 1);

			memcpy(latents_mu_smoothed, latents_mu,
				   sizeof(float) * n_latents);
			vector_plusequals(latents_mu_smoothed,
							  delta_mu, n_latents);

			float delta_cov[n_latents * n_latents];
			memcpy(delta_cov, latents_cov_smoothed_next,
				   sizeof(float) * n_latents * n_latents);
			vector_minusequals(delta_cov,
							   latents_cov_pred,
							   n_latents * n_latents);

			float G_delta[n_latents * n_latents];
			matmul(G, delta_cov, G_delta,
				   n_latents, n_latents, n_latents);
			matmul_transposed(G_delta, G,
							  latents_cov_smoothed,
							  n_latents, n_latents, n_latents);
			vector_plusequals(latents_cov_smoothed,
							  latents_cov,
							  n_latents * n_latents);

			matmul(G, latents_cov_smoothed_next,
				   latents_cov_lag1,
				   n_latents, n_latents, n_latents);

			memcpy(latents_mu_smoothed_next,
				   latents_mu_smoothed,
				   sizeof(float) * n_latents);
			memcpy(latents_cov_smoothed_next,
				   latents_cov_smoothed,
				   sizeof(float) * n_latents * n_latents);


			//Sufficient statistics
			float obs_prod_latents_mu_smoothed[n_obs*n_latents];
			//matmul_transposed(obs,latents_mu_smoothed,obs_prod_latents_mu_smoothed,n_obs,1,n_latents);
			
			vector_plusequals(latents_mu_smoothed_sum, latents_mu_smoothed, n_latents);
			vector_plusequals(latents_cov_smoothed_sum, latents_cov_smoothed, n_latents*n_latents);
			vector_plusequals(latents_cov_lag1_sum, latents_cov_lag1, n_latents*n_latents);
			//vector_plusequals(obs_prod_latents_mu_smoothed_sum, obs_prod_latents_mu_smoothed, n_obs*n_latents);
			num_datapoints++;

			//also need: sum y; sum y@y.T

			//fwrite(latents_mu_smoothed,sizeof(float),n_latents,backw_file);
			//fwrite(latents_cov_smoothed,sizeof(float),n_latents*n_latents,backw_file);
			//fwrite(latents_cov_lag1,sizeof(float),n_latents*n_latents,backw_file);

			printf("loop iter. b/steps = %d/%d\n",b,steps);
			printf("attempting ftell\n");
			printf("ftell(forw_file) = %ld\n", ftell(forw_file));
		}
		printf("loop over.\n");
	}
	printf("hi\n");

	//TODO: Return accumulated sufficient statistics
	
	for (int i = 0; i < n_latents; i++) {
		printf("%f\n",latents_mu_smoothed_sum[i]/num_datapoints);
	}
}
