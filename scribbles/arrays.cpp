int main(){
	// Create arrays on gpu
	sgl::GpuArray<int> ar1({1,2,3,4,5});
	int cpuAr[1024];
	sgl::GpuArray<int> ar2(cpuAr);
	sgl::GpuArray<int> ar3(1024);

	//v1, v2, result
	sgl::add(ar1, ar2, ar3);
	sgl::substract(ar3, ar2, ar3);
	
	ar3 = ar1 + ar2 - ar2
	ar3 = ar1.add(ar2).substract(ar2);
	
	
	ar1.getArray(); // Transfer array to host
	ar1.getElement(101); // Transfers single element to host
	ar1.setArray({1,2,3,4,5}); //overwrites existing gpu array
	ar1.setElement(1,1); //overwrites a single eleent at index 
}

zip, map, reduce, filter, 
