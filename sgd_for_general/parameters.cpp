#include "parameters.h"

C_Parameters::C_Parameters() {}

C_Parameters::C_Parameters(int _num) {
	this->element_number = _num;
	this->values = new T(_num);
	this->gradients = new T(_num);
	this->gradient_accumulations_1 = new T(_num);
	this->gradient_accumulations_2 = new T(_num);
	set_initialize_all_variable(0);
}

// Initializer
void C_Parameters::set_initialize_all_variable(T _val) {
	for (int i = 0; i < element_number; i++) {
		this->values[i] = _val;
		this->gradients[i] = _val;
		this->gradient_accumulations_1[i] = _val;
		this->gradient_accumulations_2[i] = _val;
	}
}
//
void C_Parameters::set_initialize_gradients(T _val) {
	for (int i = 0; i < element_number; i++) {
		this->gradients[i] = _val;
	}
}