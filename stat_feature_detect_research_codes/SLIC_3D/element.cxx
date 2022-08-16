#include <element.H>

using namespace std;

Element :: Element()
{
	this->x=0;
	this->y=0;
	this->z=0;
	this->var1=0;
	this->numElem=0;
}

float Element :: getDistanceEuclid(Element a, Element b, float weight,int scale)
{
	float val1=0;
	float val2=0;
    val1 = sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
    val1 = val1/(2*scale*scale*scale);
    val2 = sqrt((a.var1-b.var1)*(a.var1-b.var1));
    return weight*val1 + (1-weight)*val2;
}
