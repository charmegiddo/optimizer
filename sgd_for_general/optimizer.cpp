#include "optimizer.h"

T C_Optimizer::calculate_gradient_by_numerical_differential(C_Dataset *_data) {
	T loss = 0;

	for (int i = 0; i < _data->element_number; i++) {
		loss += fnc(_data[i], *ukv, this->batch_num);
		for (int j = 0; j < ukv->element_number; j++) {
			// finite-difference methods
			// f(x + h) - f(x - h) / 2h
			ukv->values[j] += this->delta_t;
			T deriv_a = fnc(_data[i], *ukv, this->batch_num);
			ukv->values[j] -= (this->delta_t * 2);
			T deriv_b = fnc(_data[i], *ukv, this->batch_num);
			ukv->values[j] += this->delta_t;

			T grads = (deriv_a - deriv_b) / (2 * this->delta_t);
			ukv->gradients[j] += (grads);
		}
	}

	return loss;
}

T C_Optimizer::add_weight_decay(T _w) {
	return (this->learning_rate * this->weight_decay * _w);
}

T C_Optimizer::stochastic_gradient_descend_method(T _w, T _grad) {
	return _w - (_grad * this->learning_rate);
}

void C_Optimizer::set_adam(T _beta1, T _beta2, T _epsilon) {
	this->adam_beta1 = _beta1;
	this->adam_beta2 = _beta2;
	this->adam_epsilon = _epsilon;
}

// _initial_accumulator_value: prevent divide by zero.b  Tensorflow adapted 0.1
// 	alpha = learninglate
T C_Optimizer::adam_method(T *_m, T *_v, T _w, T _grads)
{
	/*
	*_m = beta1 * (*_m) + (1 - beta1) * _grads;
	*_v = beta2 * (*_v) + (1 - beta1) * _grads * _grads;
	T m_hat = *_m / (1 - pow(beta1, _iter));
	T v_hat = *_v / (1 - pow(beta2, _iter));
	_w -= LEARNING_RATE *  m_hat / (sqrt(v_hat) + epsilon);
	*/

	//
	float lr_t = this->learning_rate * sqrt(1.0 - pow(this->adam_beta2, iter)) / (1.0 - pow(this->adam_beta1, iter));
	*_m += (1 - this->adam_beta1) * (_grads - (*_m));
	*_v += (1 - this->adam_beta2) * (pow(_grads, 2) - (*_v));
	_w -= lr_t * (*_m) / (sqrt(*_v) + this->adam_epsilon);
	_w -= add_weight_decay(_w);

	return _w;
}

void C_Optimizer::variable_update(C_Optimizer::types _type) {
	iter++;
	for (int j = 0; j < ukv->element_number; j++) {

		switch (_type)
		{
		case C_Optimizer::SGD:
			ukv->values[j] = stochastic_gradient_descend_method(ukv->values[j], ukv->gradients[j]);
			break;
		case C_Optimizer::Momentum_SGD:
			// todo
			break;
		case C_Optimizer::AdaGrad:
			// todo
			break;
		case C_Optimizer::Adam:
			ukv->values[j] = adam_method(&ukv->gradient_accumulations_1[j],
				&ukv->gradient_accumulations_2[j], ukv->values[j], ukv->gradients[j]);
			break;
		default:
			ukv->values[j] -= ukv->gradients[j] * this->learning_rate;
			break;
		}

	}

	// dispose
	ukv->set_initialize_gradients(0);
	return;
}
