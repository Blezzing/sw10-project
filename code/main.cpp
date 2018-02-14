#include <iostream>
#include <array>

#include "sgl/array.hpp"

template<typename T, long unsigned int N>
void printshit(const sgl::Array<T,N>& arr){
	std::cout << "[";
	for(long unsigned int i = 0; i < N-1; i++){
		std::cout << arr[i] << ", ";
	}
	std::cout << arr[N-1] << "]" << std::endl;
}

int main(int argc, char** argv){
	std::array<int, 10> ar0{1,2,3,4,5,6,7,8,9,0};
	
	sgl::Array<int, 10> sglar0(ar0);
	sgl::Array<int, 10> sglar1(sglar0);
	sgl::Array<int, 10> sglar2({1,2,3,4,5,6,7,8,9,10});
	sgl::Array<int, 10> sglar3;

	std::cout << "Copied from std::array:" << std::endl;
	printshit(sglar0);

	std::cout << "Copied from sgl::Array:" << std::endl;
	printshit(sglar1);

	std::cout << "Constructed from list initializer:" << std::endl;
	printshit(sglar2);

	std::cout << "Constructed without initial data:" << std::endl;
	printshit(sglar3);


	return 0;
}
