#pragma once
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include"common.h"
#include "base_value.h"

class C_Parameters: public C_B_Value {
public:
	T *gradients;
	T *gradient_accumulations_1;
	T *gradient_accumulations_2;

	C_Parameters();
	C_Parameters(int _num);
	
	// Initializer
	void set_initialize_all_variable(T _val);
	void set_initialize_gradients(T _val);

};

#endif // PARAMETERS