#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.c"

typedef struct {
	int n_obs;
	int n_latents;
	int num_datapoints;
	float *latents_mu_sum;         // [n_latents]
	float *latents_cov_sum;        // [n_latents * n_latents]
	float *latents_cov_lag1_sum;   // [n_latents * n_latents]
	float *obs_sum;                // [n_obs]
	float *obs_obs_sum;            // [n_obs * n_obs]
	float *obs_latents_sum;        // [n_obs * n_latents]
} SuffStats;

void free_suffstats(SuffStats *stats) {
	if (stats) {
		free(stats->latents_mu_sum);
		free(stats->latents_cov_sum);
		free(stats->latents_cov_lag1_sum);
		free(stats->obs_sum);
		free(stats->obs_obs_sum);
		free(stats->obs_latents_sum);
		free(stats);
	}
}

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
	//printf("%d %d %d %d %d %d\n",n_obs,n_latents,F_is_const,H_is_const,Q_is_const,R_is_const);

	//Handle constant param reading
	int F_size = n_latents*n_latents;
	int H_size = n_obs*n_latents;
	int Q_size = n_latents*n_latents;
	int R_size = n_obs*n_obs;

	// Use heap allocation for potentially large arrays
	float *F_const = malloc(F_size * sizeof(float));
	if (F_is_const) {
		fread(F_const, sizeof(float), F_size, param_file);
	}
	float *Q_const = malloc(Q_size * sizeof(float));
	if (Q_is_const) {
		fread(Q_const, sizeof(float), Q_size, param_file);
	}
	float *H_const = malloc(H_size * sizeof(float));
	if (H_is_const) {
		fread(H_const, sizeof(float), H_size, param_file);
	}
	float *R_const = malloc(R_size * sizeof(float));
	if (R_is_const) {
		fread(R_const, sizeof(float), R_size, param_file);
	}

	int param_line_size = (F_is_const ? 0 : F_size) + (H_is_const ? 0 : H_size) + (Q_is_const ? 0 : Q_size) + (R_is_const ? 0 : R_size);

	float *obs_buffer = malloc(buffer_size * n_obs * sizeof(float));
	float *param_buffer = malloc((buffer_size * param_line_size + 1) * sizeof(float));

	float *latents_mu = malloc(n_latents * sizeof(float));
	float *latents_cov = malloc(n_latents * n_latents * sizeof(float));
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

			//printf("obs ");
			for (int i = 0; i < n_obs; i++) {
				//printf("%f ",obs[i]);
			}
			//printf("\n");

			if (!latents_initialised){
				//x <- H.T@(H@H.T)^-1@obs
				float *HHT = malloc(n_obs*n_obs*sizeof(float));
				matmul_transposed(H,H,HHT,n_obs,n_latents,n_obs);
				solve(HHT, obs, n_obs, 1);
				matmul(H,obs,latents_mu,n_latents,n_obs,1);
				//P <- 0
				memset(latents_cov, 0, n_latents*n_latents*sizeof(float));
				latents_initialised = true;
				free(HHT);
			} else {
				//x <- F@x
				float *latents_mu_old = malloc(n_latents*sizeof(float));
				memcpy(latents_mu_old, latents_mu, n_latents*sizeof(float));
				matmul(F,latents_mu_old,latents_mu,n_latents,n_latents,1);
				free(latents_mu_old);
				//P <- F@P@F.T+Q
				float *FP = malloc(n_latents*n_latents*sizeof(float));
				matmul(F,latents_cov,FP,n_latents,n_latents,n_latents);
				matmul_transposed(FP,F,latents_cov,n_latents,n_latents,n_latents);
				vector_plusequals(latents_cov,Q,n_latents*n_latents);
				free(FP);
				//K <- ((H@P@H.T+R)^-1@H@P).T	#kalman gain
				float *KT = malloc(n_obs*n_latents*sizeof(float));
				matmul_transposed(H,latents_cov,KT,n_obs,n_latents,n_latents);
				float *HP = malloc(n_obs*n_latents*sizeof(float));
				matmul(H,latents_cov,HP,n_obs,n_latents,n_latents);
				float *HPHT_R = malloc(n_obs*n_obs*sizeof(float));
				matmul_transposed(HP,H,HPHT_R,n_obs,n_latents,n_obs);
				vector_plusequals(HPHT_R,R,n_obs*n_obs);
				solve(HPHT_R,KT,n_obs,n_latents);
				free(HPHT_R);
				//x <- x + K@(obs - H@x)
				float *pred = malloc(n_obs*sizeof(float));
				matmul(H,latents_mu,pred,n_obs,n_latents,1);
				vector_minusequals(obs,pred,n_obs); //modifies obs
				free(pred);
				float *latents_update = malloc(n_latents*sizeof(float));
				matmul_transposed(obs,KT,latents_update,1,n_obs,n_latents);
				vector_plusequals(latents_mu,latents_update,n_latents);
				free(latents_update);
				// Joseph form for numerical stability:
				// P = (I - K@H) @ P @ (I - K@H)^T + K @ R @ K^T
				// First compute K from KT (K = KT^T)
				float *K = malloc(n_latents*n_obs*sizeof(float));
				for (int i = 0; i < n_latents; i++) {
					for (int j = 0; j < n_obs; j++) {
						K[i*n_obs+j] = KT[j*n_latents+i];
					}
				}

				// Compute I - K@H (n_latents x n_latents)
				float *IKH = malloc(n_latents*n_latents*sizeof(float));
				matmul(K, H, IKH, n_latents, n_obs, n_latents);
				for (int i = 0; i < n_latents; i++) {
					for (int j = 0; j < n_latents; j++) {
						if (i == j) {
							IKH[i*n_latents+j] = 1.0f - IKH[i*n_latents+j];
						} else {
							IKH[i*n_latents+j] = -IKH[i*n_latents+j];
						}
					}
				}

				// Compute (I - K@H) @ P
				float *IKHP = malloc(n_latents*n_latents*sizeof(float));
				matmul(IKH, latents_cov, IKHP, n_latents, n_latents, n_latents);

				// Compute (I - K@H) @ P @ (I - K@H)^T
				float *P_joseph = malloc(n_latents*n_latents*sizeof(float));
				matmul_transposed(IKHP, IKH, P_joseph, n_latents, n_latents, n_latents);

				// Compute K @ R
				float *KR = malloc(n_latents*n_obs*sizeof(float));
				matmul(K, R, KR, n_latents, n_obs, n_obs);

				// Compute K @ R @ K^T and add to P_joseph
				float *KRKT = malloc(n_latents*n_latents*sizeof(float));
				matmul_transposed(KR, K, KRKT, n_latents, n_obs, n_latents);
				vector_plusequals(P_joseph, KRKT, n_latents*n_latents);

				// Symmetrize and copy back
				for (int i = 0; i < n_latents; i++) {
					for (int j = 0; j <= i; j++) {
						float sym = 0.5f * (P_joseph[i*n_latents+j] + P_joseph[j*n_latents+i]);
						latents_cov[i*n_latents+j] = sym;
						latents_cov[j*n_latents+i] = sym;
					}
				}

				free(K);
				free(IKH);
				free(IKHP);
				free(P_joseph);
				free(KR);
				free(KRKT);
				free(KT);
				free(HP);
			}

			//printf("latents_mu ");
			for (int i = 0; i < n_latents; i++) {
				//printf("%f ",latents_mu[i]);
			}
			//printf("\n");
			//printf("latents_cov\n");
			for (int i = 0; i < n_latents; i++) {
				for (int j = 0; j < n_latents; j++) {
					//printf("%f ",latents_cov[i*n_latents+j]);
				}
				//printf("\n");
			}
			//printf("\n");
			fwrite(latents_mu, sizeof(float), n_latents, forw_file);
			fwrite(latents_cov, sizeof(float), n_latents*n_latents, forw_file);
		}
		obs_floats_read = fread(obs_buffer, sizeof(float), n_obs * buffer_size, obs_file);
		fread(param_buffer, sizeof(float), param_line_size * buffer_size, param_file);
	} while (!(feof(obs_file) || feof(param_file)));

	// Cleanup
	free(F_const);
	free(Q_const);
	free(H_const);
	free(R_const);
	free(obs_buffer);
	free(param_buffer);
	free(latents_mu);
	free(latents_cov);
}

//Backwards step
SuffStats* write_backwards(FILE *param_file, FILE *obs_file, FILE *forw_file, FILE *backw_file, int buffer_size) {
	//printf("calling write_backwards\n");
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

	float *F_const = malloc(F_size * sizeof(float));
	float *H_const = malloc(H_size * sizeof(float));
	float *Q_const = malloc(Q_size * sizeof(float));
	float *R_const = malloc(R_size * sizeof(float));

	if (F_is_const) {
		fread(F_const, sizeof(float), F_size, param_file);
	}
	if (Q_is_const) {
		fread(Q_const, sizeof(float), Q_size, param_file);
	}
	if (H_is_const) {
		fread(H_const, sizeof(float), H_size, param_file);
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
	//printf("hello! n_latents=%d, forw_stride=%d, buffer_size=%d\n",n_latents,forw_stride,buffer_size);

	float *forw_buffer = malloc(buffer_size * forw_stride * sizeof(float));
	float *param_buffer = malloc((buffer_size * param_line_size + 1) * sizeof(float));
	float *obs_buffer = malloc(buffer_size * n_obs * sizeof(float));

	// Read obs file header and get data start position
	int obs_header[1];
	fread(obs_header, sizeof(int), 1, obs_file);
	long obs_data_start = ftell(obs_file);

	float *latents_mu = malloc(n_latents * sizeof(float));
	float *latents_cov = malloc(n_latents * n_latents * sizeof(float));

	float *latents_mu_pred = malloc(n_latents * sizeof(float));
	float *latents_cov_pred = malloc(n_latents * n_latents * sizeof(float));

	float *latents_mu_smoothed = malloc(n_latents * sizeof(float));
	float *latents_cov_smoothed = malloc(n_latents * n_latents * sizeof(float));

	float *latents_mu_smoothed_next = malloc(n_latents * sizeof(float));
	float *latents_cov_smoothed_next = malloc(n_latents * n_latents * sizeof(float));

	float *latents_cov_lag1 = malloc(n_latents * n_latents * sizeof(float));

	//printf("hey\n");
	fseek(forw_file, 0, SEEK_END);
	long end_pos = ftell(forw_file);

	fseek(forw_file, end_pos - sizeof(float) * forw_stride, SEEK_SET);
	fread(latents_mu_smoothed_next, sizeof(float), n_latents, forw_file);
	fread(latents_cov_smoothed_next, sizeof(float), n_latents * n_latents, forw_file);

	// Write last timestep directly (smoothed = filtered, no lag1_cov)
	fwrite(latents_mu_smoothed_next, sizeof(float), n_latents, backw_file);
	fwrite(latents_cov_smoothed_next, sizeof(float), n_latents * n_latents, backw_file);
	float *zeros = calloc(n_latents * n_latents, sizeof(float));
	fwrite(zeros, sizeof(float), n_latents * n_latents, backw_file);
	free(zeros);

	// Position to second-to-last timestep for the smoothing loop
	fseek(forw_file, end_pos - sizeof(float) * forw_stride, SEEK_SET);

	//Sufficient statistics - allocate struct
	SuffStats *stats = malloc(sizeof(SuffStats));
	stats->n_obs = n_obs;
	stats->n_latents = n_latents;
	stats->num_datapoints = 0;
	stats->latents_mu_sum = calloc(n_latents, sizeof(float));
	stats->latents_cov_sum = calloc(n_latents * n_latents, sizeof(float));
	stats->latents_cov_lag1_sum = calloc(n_latents * n_latents, sizeof(float));
	stats->obs_sum = calloc(n_obs, sizeof(float));
	stats->obs_obs_sum = calloc(n_obs * n_obs, sizeof(float));
	stats->obs_latents_sum = calloc(n_obs * n_latents, sizeof(float));

	// Accumulate stats for last timestep (no lag1_cov for this one)
	fseek(obs_file, 0, SEEK_END);
	long obs_end_pos = ftell(obs_file);
	fseek(obs_file, obs_end_pos - sizeof(float) * n_obs, SEEK_SET);
	float *last_obs = malloc(n_obs * sizeof(float));
	fread(last_obs, sizeof(float), n_obs, obs_file);

	vector_plusequals(stats->latents_mu_sum, latents_mu_smoothed_next, n_latents);

	// E[x_T x_T^T] = P_T + mu_T @ mu_T^T
	vector_plusequals(stats->latents_cov_sum, latents_cov_smoothed_next, n_latents * n_latents);
	float *last_mu_outer = malloc(n_latents * n_latents * sizeof(float));
	matmul_transposed(latents_mu_smoothed_next, latents_mu_smoothed_next, last_mu_outer, n_latents, 1, n_latents);
	vector_plusequals(stats->latents_cov_sum, last_mu_outer, n_latents * n_latents);
	free(last_mu_outer);

	// No lag1_cov for last timestep
	vector_plusequals(stats->obs_sum, last_obs, n_obs);

	float *last_obs_obs = malloc(n_obs * n_obs * sizeof(float));
	matmul_transposed(last_obs, last_obs, last_obs_obs, n_obs, 1, n_obs);
	vector_plusequals(stats->obs_obs_sum, last_obs_obs, n_obs * n_obs);
	free(last_obs_obs);

	float *last_obs_latents = malloc(n_obs * n_latents * sizeof(float));
	matmul_transposed(last_obs, latents_mu_smoothed_next, last_obs_latents, n_obs, 1, n_latents);
	vector_plusequals(stats->obs_latents_sum, last_obs_latents, n_obs * n_latents);
	free(last_obs_latents);
	free(last_obs);

	stats->num_datapoints++;

	// Allocate working memory for the loop (reused each iteration)
	float *FP = malloc(n_latents * n_latents * sizeof(float));
	float *tmp = malloc(n_latents * n_latents * sizeof(float));
	float *latents_cov_pred_copy = malloc(n_latents * n_latents * sizeof(float));
	float *G = malloc(n_latents * n_latents * sizeof(float));
	float *delta_mu = malloc(n_latents * sizeof(float));
	float *delta_cov = malloc(n_latents * n_latents * sizeof(float));
	float *G_delta = malloc(n_latents * n_latents * sizeof(float));
	float *mu_outer = malloc(n_latents * n_latents * sizeof(float));
	float *mu_cross = malloc(n_latents * n_latents * sizeof(float));
	float *obs_obs = malloc(n_obs * n_obs * sizeof(float));
	float *obs_latents = malloc(n_obs * n_latents * sizeof(float));

	//printf("just before do loop starts. forw_stride = %d, n_latents = %d\n",forw_stride,n_latents);
	while (true) {
		//printf("start of iter!\n");
		long cur_pos = ftell(forw_file);
		//printf("cur_pos = %ld\n",cur_pos);
		long floats_left = cur_pos / sizeof(float);
		int steps = floats_left / forw_stride;
		//printf("steps = %d\n",steps);
		if (steps > buffer_size) {
			steps = buffer_size;
		}
		if (steps <= 0) {
			break;
		}

		//printf("determining block_start_pos\n");
		long block_start_pos = cur_pos - sizeof(float) * forw_stride * steps;
		if (block_start_pos < 0) {
			break;
		}

		//printf("Attempting fseek\n");

		fseek(forw_file, block_start_pos, SEEK_SET);
		fread(forw_buffer, sizeof(float), steps * forw_stride, forw_file);
		fseek(forw_file, block_start_pos, SEEK_SET);

		long t0 = (ftell(forw_file) / sizeof(float)) / forw_stride;

		fseek(param_file,
			  param_data_start + t0 * param_line_size * sizeof(float),
			  SEEK_SET);
		fread(param_buffer, sizeof(float), steps * param_line_size, param_file);

		// Read observations for this batch
		fseek(obs_file,
			  obs_data_start + t0 * n_obs * sizeof(float),
			  SEEK_SET);
		fread(obs_buffer, sizeof(float), steps * n_obs, obs_file);

		//printf("attempting ftell\n");
		//printf("ftell(forw_file) = %ld\n", ftell(forw_file));

		for (int b = steps - 1; b >= 0; b--) {
			//printf("loop iter starting. b/steps = %d/%d\n",b,steps);
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

			matmul(F, latents_cov, FP,
				   n_latents, n_latents, n_latents);
			matmul_transposed(FP, F, latents_cov_pred,
							  n_latents, n_latents, n_latents);
			vector_plusequals(latents_cov_pred, Q,
							  n_latents * n_latents);

			// RTS gain: G = P @ F^T @ P_pred^{-1}
			// Compute as: tmp = F @ P, solve P_pred @ X = tmp, then G = X^T
			matmul(F, latents_cov, tmp,
				   n_latents, n_latents, n_latents);

			memcpy(latents_cov_pred_copy, latents_cov_pred,
				   sizeof(float) * n_latents * n_latents);
			solve(latents_cov_pred_copy, tmp,
				  n_latents, n_latents);

			// Transpose to get G = P @ F^T @ P_pred^{-1}
			for (int i = 0; i < n_latents; i++) {
				for (int j = 0; j < n_latents; j++) {
					G[i * n_latents + j] = tmp[j * n_latents + i];
				}
			}

			memcpy(latents_mu_smoothed,
				   latents_mu_smoothed_next,
				   sizeof(float) * n_latents);
			vector_minusequals(latents_mu_smoothed,
							   latents_mu_pred,
							   n_latents);

			matmul(G, latents_mu_smoothed,
				   delta_mu, n_latents, n_latents, 1);

			memcpy(latents_mu_smoothed, latents_mu,
				   sizeof(float) * n_latents);
			vector_plusequals(latents_mu_smoothed,
							  delta_mu, n_latents);

			memcpy(delta_cov, latents_cov_smoothed_next,
				   sizeof(float) * n_latents * n_latents);
			vector_minusequals(delta_cov,
							   latents_cov_pred,
							   n_latents * n_latents);

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

			//Sufficient statistics (computed BEFORE updating _next variables)
			vector_plusequals(stats->latents_mu_sum, latents_mu_smoothed, n_latents);

			// E[x_t x_t^T] = P_t + mu_t @ mu_t^T
			vector_plusequals(stats->latents_cov_sum, latents_cov_smoothed, n_latents*n_latents);
			matmul_transposed(latents_mu_smoothed, latents_mu_smoothed, mu_outer, n_latents, 1, n_latents);
			vector_plusequals(stats->latents_cov_sum, mu_outer, n_latents*n_latents);

			// E[x_{t+1} x_t^T] = Cov(x_{t+1}, x_t) + mu_{t+1} @ mu_t^T
			// Note: latents_cov_lag1 = G @ P_{t+1|T} = Cov(x_t, x_{t+1}|Y), need to transpose
			// latents_mu_smoothed_next still contains mu_{t+1|T} at this point
			float *lag1_transposed = malloc(n_latents * n_latents * sizeof(float));
			for (int i = 0; i < n_latents; i++) {
				for (int j = 0; j < n_latents; j++) {
					lag1_transposed[i * n_latents + j] = latents_cov_lag1[j * n_latents + i];
				}
			}
			vector_plusequals(stats->latents_cov_lag1_sum, lag1_transposed, n_latents*n_latents);
			free(lag1_transposed);
			matmul_transposed(latents_mu_smoothed_next, latents_mu_smoothed, mu_cross, n_latents, 1, n_latents);
			vector_plusequals(stats->latents_cov_lag1_sum, mu_cross, n_latents*n_latents);

			// NOW update _next variables for the next iteration
			memcpy(latents_mu_smoothed_next,
				   latents_mu_smoothed,
				   sizeof(float) * n_latents);
			memcpy(latents_cov_smoothed_next,
				   latents_cov_smoothed,
				   sizeof(float) * n_latents * n_latents);

			// Observation statistics
			float *obs = &obs_buffer[b * n_obs];
			vector_plusequals(stats->obs_sum, obs, n_obs);

			// obs @ obs.T (outer product)
			matmul_transposed(obs, obs, obs_obs, n_obs, 1, n_obs);
			vector_plusequals(stats->obs_obs_sum, obs_obs, n_obs * n_obs);

			// obs @ latents_mu_smoothed.T (cross term)
			matmul_transposed(obs, latents_mu_smoothed, obs_latents, n_obs, 1, n_latents);
			vector_plusequals(stats->obs_latents_sum, obs_latents, n_obs * n_latents);

			stats->num_datapoints++;

			fwrite(latents_mu_smoothed,sizeof(float),n_latents,backw_file);
			fwrite(latents_cov_smoothed,sizeof(float),n_latents*n_latents,backw_file);
			fwrite(latents_cov_lag1,sizeof(float),n_latents*n_latents,backw_file);

			//printf("loop iter. b/steps = %d/%d\n",b,steps);
			//printf("attempting ftell\n");
			//printf("ftell(forw_file) = %ld\n", ftell(forw_file));
		}
		//printf("loop over.\n");
	}

	// Cleanup
	free(FP);
	free(tmp);
	free(latents_cov_pred_copy);
	free(G);
	free(delta_mu);
	free(delta_cov);
	free(G_delta);
	free(mu_outer);
	free(mu_cross);
	free(obs_obs);
	free(obs_latents);
	free(F_const);
	free(H_const);
	free(Q_const);
	free(R_const);
	free(forw_buffer);
	free(param_buffer);
	free(obs_buffer);
	free(latents_mu);
	free(latents_cov);
	free(latents_mu_pred);
	free(latents_cov_pred);
	free(latents_mu_smoothed);
	free(latents_cov_smoothed);
	free(latents_mu_smoothed_next);
	free(latents_cov_smoothed_next);
	free(latents_cov_lag1);

	return stats;
}
