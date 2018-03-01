#ifndef __BASE_VALUE_H__
#define __BASE_VALUE_H__

#include "common.h"

class C_B_Value {
public:
	T *values;
	int element_number;

	C_B_Value();
	virtual void disp();
};

#endif // BASE_VALUE