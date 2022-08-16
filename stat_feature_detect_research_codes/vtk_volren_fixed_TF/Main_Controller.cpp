#include "cpp_headers.h"
#include <GL/gl.h> 
#include <GL/glut.h>
#include <GL/glui.h>

using namespace std;

int main_window;
int split_screen_by = 4;
int width=600; 
int height=600;
float vertSpan=height/(float)split_screen_by;
float AxisStartX=0;
float AxisStartY=vertSpan;
int span;
int numCtrlPts=6;
int xx,yy;
float maxi = 12.1394; //TODO
float mini = 0.0070086; //TODO

float **opacityCtrlPointsX;
float **opacityCtrlPointsY;

void print_bitmap_string(void* font, char* s)
{
	if (s && strlen(s)) {
		while (*s) {
			glutBitmapCharacter(font, *s);
			s++;
		}
	}
}

void updatePoints(float finalposX,float finalposY,int x, int y)
{
	
	if(finalposX<0)
		finalposX=0;

	if(finalposX>width)
		finalposX=width;

	if(x==0)
	{	
		if(finalposY < 0)
			finalposY = 0;

		if(finalposY>vertSpan)
			finalposY=vertSpan;
	}
	if(x==1)
	{	
		if(finalposY < vertSpan)
			finalposY = vertSpan;

		if(finalposY>2*vertSpan)
			finalposY=2*vertSpan;
	}
	if(x==2)
	{	
		if(finalposY < 2*vertSpan)
			finalposY = 2*vertSpan;

		if(finalposY>3*vertSpan)
			finalposY=3*vertSpan;
	}
	if(x==3)
	{	
		if(finalposY < 3*vertSpan)
			finalposY = 3*vertSpan;

		if(finalposY>height)
			finalposY=height;
	}

	opacityCtrlPointsX[x][y] = finalposX;
	opacityCtrlPointsY[x][y] = finalposY;
}

void mouseButton(int button, int state, int x, int y) 
{
	float initposX,initposY,finalposX,finalposY;	

	//write code here
	if(button == GLUT_LEFT_BUTTON)
	{
		if(state == GLUT_DOWN)
			{
				for(int i=0;i<split_screen_by;i++)
				{
					for(int j=0;j<numCtrlPts;j++)
					{
						if((fabs(x-opacityCtrlPointsX[i][j])<8) && (fabs(height - y - opacityCtrlPointsY[i][j])<8))
							{
								xx = i;
								yy = j;	
							}
					}
				}

			}
		if(state == GLUT_UP)
			{
				finalposX = x;
				finalposY = height - y;
				updatePoints(finalposX,finalposY,xx,yy);
			}
	}

	ofstream writeFile1("x_vals.txt"); 
	ofstream writeFile2("y_vals.txt");

	for(int i=0;i<split_screen_by;i++)
		{
			for(int j=0;j<numCtrlPts;j++)
				{
					writeFile1<<ceil(((opacityCtrlPointsX[i][j]-AxisStartX)/span)*255)<<" ";
					writeFile2<<(opacityCtrlPointsY[i][j]-i*vertSpan)/(float)vertSpan<<" "; // for converting range into (0-1)					
				}

				writeFile1<<endl;
				writeFile2<<endl;
		}

	writeFile1.close();	
	writeFile2.close();	
	
	glutPostRedisplay();
}

void myKey(unsigned char key, int x, int y)
{
	if(key=='w')
	{
		ofstream writeFile1("x_vals.txt"); 
		ofstream writeFile2("y_vals.txt");

			for(int i=0;i<split_screen_by;i++)
			{
				for(int j=0;j<numCtrlPts;j++)
				{
					writeFile1<<ceil(((opacityCtrlPointsX[i][j]-AxisStartX)/span)*255)<<" ";
					writeFile2<<(opacityCtrlPointsY[i][j]-i*vertSpan)/(float)vertSpan<<" "; // for converting range into (0-1)
				}

				writeFile1<<endl;
				writeFile2<<endl;
			}

			writeFile1.close();	
			writeFile2.close();	

	}
}

void display()
{
	glClearColor(0.2,0.3,0.3,1);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POINT_SMOOTH);

	//Set up 2D projection
	glMatrixMode(GL_MODELVIEW); 
	glLoadIdentity(); 
	glMatrixMode(GL_PROJECTION); 
	glLoadIdentity();
	
	gluOrtho2D(0,width,0,height);
	
	//Draw Separator line
	glLineWidth(4);
	glColor3f(1,1,1);
	glBegin(GL_LINES);
	for(int j=0;j<numCtrlPts-1;j++)
		{
			glVertex2f(0,(height/split_screen_by)*(j+1));
			glVertex2f(width,(height/split_screen_by)*(j+1));
		}
	glEnd();	

	//Draw Connecting Lines 
	glLineWidth(3);
	glBegin(GL_LINES);
	for(int i=0;i<split_screen_by;i++)
	{
		if(i==3)
			glColor3f(1,0,0);
		else if(i==2)
			glColor3f(0,1,0);
		else if(i==1)
			glColor3f(0,0,1);
		else 
			glColor3f(0.8,0.8,0.8);

		for(int j=0;j<numCtrlPts-1;j++)
		{
			glVertex2f(opacityCtrlPointsX[i][j],opacityCtrlPointsY[i][j]);
			glVertex2f(opacityCtrlPointsX[i][j+1],opacityCtrlPointsY[i][j+1]);
		}
	}
	glEnd();

	//Draw Control Points
	glPointSize(15);	
	glBegin(GL_POINTS);
	for(int i=0;i<split_screen_by;i++)
	{
		if(i==3)
			glColor3f(1,0,0);
		else if(i==2)
			glColor3f(0,1,0);
		else if(i==1)
			glColor3f(0,0,1);
		else 
			glColor3f(0.8,0.8,0.8);

		for(int j=0;j<numCtrlPts;j++)
		{
			glVertex2f(opacityCtrlPointsX[i][j],opacityCtrlPointsY[i][j]);
		}
	}
	glEnd();

	glColor3f(1,1,1);
	char str[50];
	float val,val1;
	for(int i=0;i<split_screen_by;i++)
	{
		for(int j=0;j<=10;j++)
		{
			stringstream ss; 
			//ss <<j/(float)10; // old to show numbers between 0-1 
			val1 = j/(float)10;
			val = mini + val1*(maxi-mini);
			ss<<val;
			strcpy(str,ss.str().c_str());
			glRasterPos2f(AxisStartX + j*(width/10),5 + i*vertSpan);
			print_bitmap_string(GLUT_BITMAP_HELVETICA_12,str);
		}
	}		
	
	//Swap Buffer at the end!
	glutSwapBuffers();
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv); 
	glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE); 
	glutInitWindowSize(width,height);
	main_window = glutCreateWindow("Transfer Function Editor"); 
	span = width - 2*AxisStartX;

	opacityCtrlPointsX = (float **)malloc(split_screen_by*sizeof(float *));
	for(int i=0;i<split_screen_by;i++)
		opacityCtrlPointsX[i] = (float *)malloc(numCtrlPts*sizeof(float));

	opacityCtrlPointsY = (float **)malloc(split_screen_by*sizeof(float *));
	for(int i=0;i<split_screen_by;i++)
		opacityCtrlPointsY[i] = (float *)malloc(numCtrlPts*sizeof(float));
	
	//Initialize control points locations at the center of each cells	
	for(int i=0;i<split_screen_by;i++)
	{
		for(int j=0;j<numCtrlPts;j++)
			{
				opacityCtrlPointsX[i][j] = AxisStartX + (span/(numCtrlPts-1))*j;	
				opacityCtrlPointsY[i][j] = AxisStartY +  i*vertSpan;			
			}		
	}	

	glutMouseFunc(mouseButton);
	glutKeyboardFunc(myKey); 
	glutDisplayFunc(display); 

	glutMainLoop();

	free(opacityCtrlPointsX);
	free(opacityCtrlPointsY);

	return 0;
}