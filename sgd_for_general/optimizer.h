#pragma once
#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include "common.h"
#include "parameters.h"
#include "dataset.h"

class C_Optimizer {
private:
	T adam_beta1;
	T adam_beta2;
	T adam_epsilon;

public:
	enum types
	{
		SGD = 0,
		Momentum_SGD,
		AdaGrad,
		Adam
	};
	function<T(C_Dataset, C_Parameters, int)> fnc;
	C_Parameters *ukv;

	int batch_num;
	int iter;
	T delta_t;
	T learning_rate;
	T weight_decay;


	C_Optimizer() {
	}

	C_Optimizer(function<T(C_Dataset, C_Parameters, int)> _fnc, C_Parameters *_ukv, int _batch_num, T _delta_t, T learning_rate) {
		this->fnc = _fnc;
		this->ukv = _ukv;
		this->batch_num = _batch_num;
		this->delta_t = _delta_t;
		this->learning_rate = learning_rate;
		this->iter = 0;
		this->weight_decay = 0;
		this->adam_beta1 = 0.9;
		this->adam_beta2 = 0.999;

		this->adam_epsilon = 1e-8;
	}


	T calculate_gradient_by_numerical_differential(C_Dataset *_data);
	T add_weight_decay(T _w);
	T stochastic_gradient_descend_method(T _w, T _grad);
	void set_adam(T _beta1 = 0.9, T _beta2 = 0.999, T _epsilon = 1e-8);

	// _initial_accumulator_value: prevent divide by zero.b  Tensorflow adapted 0.1
	// 	alpha = learninglate
	T adam_method(T *_m, T *_v, T _w, T _grads);
	void variable_update(C_Optimizer::types _type);
};

#endif  // OPTIMIZER_H
