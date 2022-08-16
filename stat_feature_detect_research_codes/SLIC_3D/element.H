#ifndef ELEMENT
#define ELEMENT

#include <cpp_headers.H>

class Element
{
	
public:

	Element();
    float x;
    float y;
    float z;
    float var1;
    int numElem;

    float getDistanceEuclid(Element,Element,float,int);
};

#endif 
