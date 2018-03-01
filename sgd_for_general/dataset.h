#pragma once
#ifndef __DATASET_H__
#define __DATASET_H__
#include "common.h"
#include "base_value.h"


class C_Dataset : public C_B_Value {
public:
	T output;
	C_Dataset(int _num);
	C_Dataset();
	void set_assets(T *_data, T _output);
	virtual void disp();

};
#endif // DATASET_H