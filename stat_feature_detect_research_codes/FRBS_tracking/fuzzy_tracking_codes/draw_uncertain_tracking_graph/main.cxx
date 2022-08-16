////////////////////////////////////////////////////
// Soumya Dutta
// The Ohio State University
// CSE Department/CCS-7 LANL
// Autumn 2015: Modified July 2020
// The Code uses openGL,GLUT,GLM
////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <sstream>
#include <cpp_headers.h>
#include <glm_headers.h>
#include <opengl_headers.h>

using namespace std;
using namespace glm;

int main_window;
int width=2000;
int height=600;
int axisStartX=75;
int axisStartY=100;
int time_steps;
int lowerBoundY,upperBoundY;
float lineThickness=1.0;
int axisDiffX;
int basePointSize = 3;
int bb=0;
float val;
float color[3];
vector< vector <float> > graph_info;
vector< vector <float> > normalized_graph_info;
vector<int> id_list;
int initStep=19000;

vector<float> split(string str, string sep)
{
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    vector<float> arr;
    current=strtok(cstr,sep.c_str());
    while(current != NULL){
        arr.push_back(atof(current));
        current=strtok(NULL, sep.c_str());
    }
    return arr;
}

vector<float> normalize_vector(vector<float> in)
{
    float max = in[0];
    float min = in[0];

    if(in.size()>1)
    {

        for(int i=0;i<in.size();i++)
        {
            if(max<in[i])
                max = in[i];

            if(min>in[i])
                min=in[i];
        }

        for(int i=0;i<in.size();i++)
        {
            in[i] = (in[i]-min)/(max-min);
        }
    }

    return in;
}

int get_matched_ids(vector<float> in)
{
    float max = in[0];
    int id = 0;

    if(in.size()>1)
    {
        for(int i=0;i<in.size();i++)
        {
            if(max<in[i])
            {
                max = in[i];
                id = i;
            }
        }
    }

    return id;
}

void GroundColorMix(float* color, float x, float min, float max)
{
    /*
    * Red = 0
    * Green = 1
    * Blue = 2
    */
    float posSlope = (max-min)/60;
    float negSlope = (min-max)/60;

    if( x < 60 )
    {
        color[0] = max;
        color[1] = posSlope*x+min;
        color[2] = min;
        return;
    }
    else if ( x < 120 )
    {
        color[0] = negSlope*x+2*max+min;
        color[1] = max;
        color[2] = min;
        return;
    }
    else if ( x < 180  )
    {
        color[0] = min;
        color[1] = max;
        color[2] = posSlope*x-2*max+min;
        return;
    }
    else if ( x < 240  )
    {
        color[0] = min;
        color[1] = negSlope*x+4*max+min;
        color[2] = max;
        return;
    }
    else if ( x < 300  )
    {
        color[0] = posSlope*x-4*max+min;
        color[1] = min;
        color[2] = max;
        return;
    }
    else
    {
        color[0] = max;
        color[1] = min;
        color[2] = negSlope*x+6*max;
        return;
    }
}

void mouseButton(int button, int state, int x, int y) 
{
    y=height-y;
    int ypos = 0;
    int xpos =0;

    if(state == GLUT_UP)
    {
        for(int i=0;i<time_steps;i++)
        {
            xpos = axisStartX+(i+1)*axisDiffX;
            
            for(int j=0;j<=normalized_graph_info[i].size()-1;j++)
            {
                if(normalized_graph_info[i].size()>1)
                    ypos = axisStartY + j*(upperBoundY-lowerBoundY)/(normalized_graph_info[i].size()-1);
                else
                    ypos = axisStartY;   

                // //if(abs(xpos-x) < 5 && abs(ypos-y) < 5)
                // if(i==10 && j==2)
                //     cout<<"position of cursor: "<<xpos<<" "<<ypos<<" "<<x<<" "<<y<<" "<<normalized_graph_info[i][j]<<endl;                
            }   

        }
    }
}

void mykey(unsigned char key, int x, int y)
{
    if (key == 'q')
        exit(0);
}

void renderStrokeFontString(
        float x,
        float y,
        float z,
        void *font,
        char *string) 
{

    char *c;
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(3.5);
    glPushMatrix();
    glTranslatef(x,y,z);
    glScalef(0.15,0.15,0.15);
    for (c=string; *c != '\0'; c++) {
        glutStrokeCharacter(font, *c);
    }

    glPopMatrix();
}

// GLUT_BITMAP_8_BY_13
// GLUT_BITMAP_9_BY_15
// GLUT_BITMAP_TIMES_ROMAN_10
// GLUT_BITMAP_TIMES_ROMAN_24
// GLUT_BITMAP_HELVETICA_10
// GLUT_BITMAP_HELVETICA_12
// GLUT_BITMAP_HELVETICA_18 
void drawBitmapText(char *string,float x,float y) 
{   
    ////Old technique, works but fonts are small
    // char *c;
    // glRasterPos2f(x,y);
    // for (c=string; *c != '\0'; c++) 
    // {
    //     glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *c);
    // }

    //mew approach
    renderStrokeFontString(x,y,1,GLUT_STROKE_ROMAN,string);
}

void draw_init_feature()
{
    basePointSize=40;    
    bb = ceil(basePointSize*0.2);

    glColor4f(0,0,0,1.0);
    glPointSize(basePointSize+bb);
    glBegin(GL_POINTS);
    glVertex2f(axisStartX,height/2);
    glEnd();

    glColor4f(1.0,0,0,1.0);
    glPointSize(basePointSize);
    glBegin(GL_POINTS);
    glVertex2f(axisStartX,height/2);
    glEnd();

    glColor4f(0.0,0,0,1.0);
    char* name = "Starting Feature";
    drawBitmapText(name,axisStartX-60,height/2+40);
}

void motionCallBack(int x, int y)
{
    y=height-y;
    int xpos,ypos;

    //cout<<"position of cursor: "<<x<<" "<<y<<endl;

    for(int i=0;i<time_steps;i++)
    {
        int xpos = axisStartX+(i+1)*axisDiffX;
        int ypos = 0;
        for(int j=0;j<=normalized_graph_info[i].size()-1;j++)
        {
            if(normalized_graph_info[i].size()>1)
                ypos = axisStartY + j*(upperBoundY-lowerBoundY)/(normalized_graph_info[i].size()-1);
            else
                ypos = axisStartY;   

            if(abs(xpos-x) < 1 && abs(ypos-y) < 1)
                cout<<"position of cursor: "<<x<<" "<<y<<endl;  

            //cout<<xpos<<" "<<ypos<<endl;               
        }
    }

    glutPostRedisplay();
}

void display()
{
    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POINT_SMOOTH);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    float xpos1,xpos2,ypos1,ypos2;

    //Set up 2D projection
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,width,0,height);

    //Draw Vertical lines
    lineThickness=1.0;
    glLineWidth(lineThickness);
    glBegin(GL_LINES);
    glColor4f(0,0,0,0.2);
    for(int i=1;i<=time_steps;i++)
    {
       glVertex2f(axisStartX+i*axisDiffX,lowerBoundY);
       glVertex2f(axisStartX+i*axisDiffX,upperBoundY);
    }
    glEnd();

    //draw small horizontal segments
    lineThickness=2.0;
    glLineWidth(lineThickness);
    glBegin(GL_LINES);
    glColor4f(0,0,0,1);
    for(int i=1;i<=time_steps;i++)
    {
       glVertex2f(axisStartX+i*axisDiffX-12,lowerBoundY);
       glVertex2f(axisStartX+i*axisDiffX+12,lowerBoundY);
    }
    glEnd();

    glBegin(GL_LINES);
    glColor4f(0,0,0,1);
    for(int i=1;i<=time_steps;i++)
    {
       glVertex2f(axisStartX+i*axisDiffX-12,upperBoundY);
       glVertex2f(axisStartX+i*axisDiffX+12,upperBoundY);
    }
    glEnd();

    //Draw time step values
    glColor4f(0,0,0,1);
    for(int i=1;i<=time_steps;i++)
    {
        xpos1 = axisStartX+i*axisDiffX-12;
        ypos1 = lowerBoundY-40;
        stringstream ss; 
        ss << initStep+i*400;
        //char* val = ss.str();
        char *val = new char[ss.str().length() + 1];
        std::strcpy(val, ss.str().c_str());
        drawBitmapText(val,xpos1,ypos1);
    }

    //draw connecting line
    /////////////////////////////////////   
    for(int i=0;i<time_steps;i++)
    {
        for(int j=0;j<=normalized_graph_info[i].size()-1;j++)
        {
            val = normalized_graph_info[i][j];
            if (val>0.01)
            {    
                glPushAttrib(GL_ENABLE_BIT);  // for dotted line
                glLineStipple(5, 0xAAAA); // for dotted line
                glEnable(GL_LINE_STIPPLE); // for dotted line
                if(i==0)
                {
                    xpos1 = axisStartX;
                    ypos1 = height/2;
                    
                    xpos2 = axisStartX + (i+1)*axisDiffX;

                    if(normalized_graph_info[i].size()>1)
                        ypos2 = axisStartY + id_list[i]*(upperBoundY-lowerBoundY)/(normalized_graph_info[i].size()-1);
                    else
                        ypos2 = axisStartY;

                    lineThickness=3;
                    glLineWidth(lineThickness);
                    glBegin(GL_LINES);
                    glColor4f(0,0,0,0.2);
                    glVertex2f(xpos1,ypos1);
                    glVertex2f(xpos2,ypos2);
                    glEnd();
                }
                else
                {
                    xpos1 = axisStartX + i*axisDiffX;
                    xpos2 = axisStartX + (i+1)*axisDiffX;

                    if(normalized_graph_info[i-1].size()>1)
                            ypos1 = axisStartY + id_list[i-1]*(upperBoundY-lowerBoundY)/(normalized_graph_info[i-1].size()-1);
                        else
                            ypos1 = axisStartY;

                    if(normalized_graph_info[i].size()>1)                
                        ypos2 = axisStartY + j*(upperBoundY-lowerBoundY)/(normalized_graph_info[i].size()-1);
                    else
                        ypos2 = axisStartY;

                    lineThickness=3;
                    glLineWidth(lineThickness);
                    glBegin(GL_LINES);
                    glColor4f(0,0,0,0.3);
                    glVertex2f(xpos1,ypos1);
                    glVertex2f(xpos2,ypos2);
                    glEnd();                
                }
                glPopAttrib(); // for dotted line

                //redraw the tracked line
                if(i==0)
                {
                    xpos1 = axisStartX;
                    ypos1 = height/2;
                    
                    xpos2 = axisStartX + (i+1)*axisDiffX;

                    if(normalized_graph_info[i].size()>1)
                    {
                        ypos2 = axisStartY + id_list[i]*(upperBoundY-lowerBoundY)/(normalized_graph_info[i].size()-1);
                    }
                    else
                    {
                        ypos2 = axisStartY;
                    }

                    lineThickness=5;
                    glLineWidth(lineThickness);
                    glBegin(GL_LINES);
                    glColor4f(1,0,0,1.0);
                    glVertex2f(xpos1,ypos1);
                    glVertex2f(xpos2,ypos2);
                    glEnd();
                }
                else
                {
                    if(id_list[i]==j)
                    {
                        xpos1 = axisStartX + i*axisDiffX;
                        xpos2 = axisStartX + (i+1)*axisDiffX;

                        if(normalized_graph_info[i-1].size()>1)
                                ypos1 = axisStartY + id_list[i-1]*(upperBoundY-lowerBoundY)/(normalized_graph_info[i-1].size()-1);
                            else
                                ypos1 = axisStartY;

                        if(normalized_graph_info[i].size()>1)
                        {                        
                            ypos2 = axisStartY + j*(upperBoundY-lowerBoundY)/(normalized_graph_info[i].size()-1);
                        }
                        else
                        {
                            ypos2 = axisStartY;
                        }

                        lineThickness=5;
                        glLineWidth(lineThickness);
                        glBegin(GL_LINES);
                        glColor4f(1,0,0,1.0);
                        glVertex2f(xpos1,ypos1);
                        glVertex2f(xpos2,ypos2);
                        glEnd();
                    }
                }
            }
        }
    }

    draw_init_feature();

    //draw feature points
    for(int i=0;i<time_steps;i++)
    {
        int xpos = axisStartX+(i+1)*axisDiffX;
        int ypos = 0;
        for(int j=0;j<=normalized_graph_info[i].size()-1;j++)
        {

            val = normalized_graph_info[i][j];
            
            if(val>0.01)
            {
                if (val < 0.3) // 0.3 for vortex data
                    basePointSize = 12;
                else
                    basePointSize = 45*val; // 45 for vortex data

                //this is when confidence value is zero
                if(basePointSize==0)
                    basePointSize=10.0;

                glPointSize(basePointSize);

                //clamp val to fix range between 0 and 1 for color lookup
                if(val>1.0)
                    val = 1.000;
                if(val<=0)
                    val = 0.0;

                GroundColorMix(color,240.0 - val*240.0,0,1);            

                if(normalized_graph_info[i].size()>1)
                    ypos = axisStartY + j*(upperBoundY-lowerBoundY)/(normalized_graph_info[i].size()-1);
                else
                    ypos = axisStartY;

                bb = ceil(basePointSize*0.2);
                glColor4f(0,0,0,1.0);
                glPointSize(basePointSize+bb);
                glBegin(GL_POINTS);
                glVertex2f(xpos,ypos);
                glEnd();

                glColor4f(color[0],color[1],color[2],1.0);
                glPointSize(basePointSize);
                glBegin(GL_POINTS);
                glVertex2f(xpos,ypos);
                glEnd();

                //show conf values
                glColor4f(0,0,0,1);
                stringstream vv;
                vv<<setprecision(3)<<val;
                char *vall = new char[vv.str().length() + 1];
                std::strcpy(vall, vv.str().c_str());
                drawBitmapText(vall,xpos-9,ypos+15);

            }
        }
    }

    glColor4f(0,0,0,1);
    drawBitmapText("Time steps: ",axisStartX-60,axisStartY-40);

    glutSwapBuffers();
}

int main(int argc, char **argv)
{
    /* initialize random seed: */
    srand (time(NULL));

    //Read the parameters of the trained fuzzy rule based system
    //////////////////////////////////////////////////////////////

    ifstream readoutFis;
    //readoutFis.open("/Users/sdutta/Codes/fuzzy_rule_based_tracking/fuzzy_tracking_codes/fuzzy_rule_based_tracking/build/graphfile.txt");
    readoutFis.open("/Users/sdutta/Codes/fuzzy_rule_based_tracking/fuzzy_tracking_codes/fuzzy_rule_based_tracking_mfix/build/graphfile.txt");

    string line;

    while(!readoutFis.eof())
    {
        getline(readoutFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, ",");

            if(v.size()>0)
            {
                graph_info.push_back(v);
            }
        }
    }

    time_steps = graph_info.size();

    ///print graph file
    for(int ii=0; ii<graph_info.size();ii++)
    {
        //normalized_graph_info.push_back(normalize_vector(graph_info[ii]));
        normalized_graph_info.push_back(graph_info[ii]);
    }

    for(int ii=0; ii<graph_info.size();ii++)
    {
        id_list.push_back(get_matched_ids(graph_info[ii]));
    }

    lowerBoundY = axisStartY;
    upperBoundY=height-axisStartY;
    int totalDiffX = width - 2*axisStartX;
    axisDiffX = totalDiffX/time_steps;

    //OpenGL Calls
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE);

    glutInitWindowSize(width,height);
    main_window = glutCreateWindow("Confidence-based Tracking Dynamics Graph");

    glutKeyboardFunc(mykey);
    glutMouseFunc(mouseButton);
    glutPassiveMotionFunc(motionCallBack);
    glutDisplayFunc(display);

    glutMainLoop();

    return 0;
}
