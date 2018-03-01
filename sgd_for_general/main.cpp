#include <stdio.h>
#include <iostream>
#include <vector>
#include <functional>
#include <stdlib.h> 

#include "common.h"
#include "dataset.h"
#include "parameters.h"
#include "optimizer.h"


#define BATCH_NUM 10
#define EPOCH_NUM 10000

/////////////////////////////////////////////////////////////////////
/////////////////////////// Your method /////////////////////////////
/////////////////////////////////////////////////////////////////////
T least_square(T _ans, T _est) {
	return pow((_ans - _est), 2) * 0.5;
}

T generate_training_data(T _a, T _b) {
	// Unknown variables: 10, 3
	// _a^2 * c * d + _b^2 * d
	// c = 10, d = 3
	return (_a * 10 * _a * 3) + (_b * 3 * _b * 3);
}

T func(C_Dataset _data, C_Parameters _ukv, int _batch_num) {
	// Estimate
	T _a = _data.values[0];
	T _b = _data.values[1];
	T _c = _ukv.values[0];
	T _d = _ukv.values[1];
	T _est = (_a * _a * _c * _d) + (_b * _b * _d * _d);

	// Loss
	T loss = least_square(_data.output, _est);

	return loss / _batch_num;
}

/////////////////////////////////////////////////////////////////////
///////////////////////////  Entrypoint /////////////////////////////
/////////////////////////////////////////////////////////////////////
int main() {
	// Search gradients for unknowns
	C_Parameters *ukv = new C_Parameters(2); ukv->values[0] = 0.5; ukv->values[1] = 0.5;
	C_Optimizer *optimizer = new C_Optimizer(func, ukv, BATCH_NUM, 1e-5, 1e-2);
	optimizer->set_adam(0.9, 0.999, 1e-8);

	// Generate dataset
	vector<vector<C_Dataset>> minibatch;
	for (int batch = 0; batch < EPOCH_NUM; batch++) {
		vector<C_Dataset> dataset;
		T tmp[2];
		for (int i = 0; i < BATCH_NUM; i++) {
			dataset.push_back(2);
			for (int j = 0; j < dataset.back().element_number; j++) {
				tmp[j] = (float)(rand() % 100) / 100;
			}
			T out_tmp = generate_training_data(tmp[0], tmp[1]);
			dataset.back().set_assets(tmp, out_tmp);
		}
		// Register each (mini-) batch
		minibatch.push_back(dataset);
	}

	// draw training data;
	/*for (int i = 0; i < minibatch.size(); i++) {
	vector<C_Dataset> dataset = minibatch[i];
	for (int j = 0; j < dataset.size(); j++) {
	dataset[j].disp();
	}
	}*/

	// training
	for (int i = 0; i < minibatch.size(); i++) {
		vector<C_Dataset> dataset = minibatch[i];
		T loss = optimizer->calculate_gradient_by_numerical_differential(&dataset.front());
		optimizer->variable_update(C_Optimizer::types::Adam);
		cout << optimizer->iter; cout << " loss: " << loss << " ";  ukv->disp();
		if (i % 500 == 0) getchar();
	}
}
