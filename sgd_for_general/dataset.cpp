#include "dataset.h"

C_Dataset::C_Dataset() {
}


C_Dataset::C_Dataset(int _num) {
	this->element_number = _num;
	this->values = new T[_num];
	this->output = 0;
}


void C_Dataset::disp() {
	for (int i = 0; i < element_number; i++) {
		printf("[%d] %f, ", i, values[i]);
	}
	printf("[out] %f", output);
	cout << endl;
}

void C_Dataset::set_assets(T *_data, T _output) {
	for (int i = 0; i < this->element_number; i++) this->values[i] = _data[i];
	this->output = _output;
	return;
}