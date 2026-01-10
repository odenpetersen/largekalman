FILE* open_file_write(const char *path) {
	FILE *f = fopen(path, "wb+");
	return f;
}

void* open_file_read(const char *path) {
	FILE *f = fopen(path, "rb");
	return f;
}

void close_file(FILE *f) {
	if (f) fclose(f);
}

void free_buffer(void *p) {
	free(p);
}

void write_ints(const int *ints, int n, void *handle) {
	fwrite(ints, sizeof(int), n, handle);
}

void write_floats(const float *floats, int n, void *handle) {
	fwrite(floats, sizeof(float), n, handle);
}

int *read_ints(int n, void *handle) {
	int *ints = malloc(sizeof(int) * n);
	fread(ints, sizeof(int), n, handle);
	return ints;
}

float *read_floats(int n, void *handle) {
	float *floats = malloc(sizeof(float) * n);
	fread(floats, sizeof(float), n, handle);
	return floats;
}

float *read_floats_backwards(int n, void *handle) {
	float *floats = malloc(sizeof(float) * n);
	fseek(handle, n*sizeof(float), SEEK_CUR);
	fread(floats, sizeof(float), n, handle);
	return floats;
}

void matmul(float *A, float *B, float *res, int height_res, int common_dim, int width_res) {
	for (int i = 0; i < height_res; i++) {
		for (int j = 0; j < width_res; j++) {
			float sum = 0;
			for (int k = 0; k < common_dim; k++) {
				sum += A[i*common_dim+k]*B[k*width_res+j];
			}
			res[i*width_res+j] = sum;
		}
	}
}

void matmul_transposed(float *A, float *BT, float *res, int height_res, int common_dim, int width_res) {
	for (int i = 0; i < height_res; i++) {
		for (int j = 0; j < width_res; j++) {
			float sum = 0;
			for (int k = 0; k < common_dim; k++) {
				sum += A[i*common_dim+k]*BT[j*common_dim+k];
			}
			res[i*width_res+j] = sum;
		}
	}
}

void vector_plusequals(float *A, float *B, int n) {
	for (int i = 0; i < n; i++) {
		A[i] += B[i];
	}
}

void vector_minusequals(float *A, float *B, int n) {
	for (int i = 0; i < n; i++) {
		A[i] -= B[i];
	}
}

void solve(float *A, float *B, int n, int m) {
	// Gaussian elimination
	for (int k = 0; k < n; k++) {
		for (int i = k+1; i < n; i++) {
			float factor = A[i*n+k] / A[k*n+k];
			for (int j = k; j < n; j++) {
				A[i*n+j] -= factor * A[k*n+j];
			}
	
			for (int j = 0; j < m; j++) {
				B[i*m+j] -= factor * B[k*m+j];
			}
		}
	}

	// Back-substitution
	for (int k = n-1; k >= 0; k--) {
		for (int j = 0; j < m; j++) {
			B[k*m+j] /= A[k*n+k];
		}
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < m; j++) {
				B[i*m+j] -= A[i*n+k] * B[k*m+j];
			}
		}
	}
}
