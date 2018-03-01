#include "base_value.h"

C_B_Value::C_B_Value() {
}
void C_B_Value::disp() {
	for (int i = 0; i < element_number; i++) {
		printf("[%d] %f, ", i, values[i]);
	}
	cout << endl;
}