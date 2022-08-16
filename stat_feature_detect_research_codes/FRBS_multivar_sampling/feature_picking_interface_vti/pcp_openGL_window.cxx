/////////////////////////////////////////////////////
//Soumya Dutta
//OSU CSE 2017
/////////////////////////////////////////////////////
#include <GL/gl.h>
#include <GL/glu.h>
#include <pcp_openGL_window.h>
#include <QtOpenGL/QGLFormat>
#include <QtOpenGL/QGLPixelBuffer>

double *colorval;
float maxvals[3];
float minvals[3];

vector<float> pcp_openGL_window :: split(string str, string sep)
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

/*
* Computes the color gradiant
* color: the output vector
* x: the gradiant (beetween 0 and 360)
* min and max: variation of the RGB channels (Move3D 0 -> 1)
*/
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

void pcp_openGL_window :: draw_colormap(int startX,int startY)
{
    int steps=200;
    QFont font;

    //draw colormap
    ///////////////////////////////////////////////
    for(int i=0;i<=steps;i++)
    {
        float val = i/(float)steps;
        colorval = this->colorTF->GetColor(1-val); // reverse colormap
        glColor4f(colorval[0],colorval[1],colorval[2],1.0);

        glLineWidth(2.0);
        glBegin(GL_LINES);
        glVertex2f(startX,startY+i); // stripWidth because the width for each timestep is stripWidth pixels
        glVertex2f(startX+10,startY+i);
        glEnd();
    }

    ///cmap label
    glColor4f(0,0,0,1);
    font.setPointSize(11);
    QString text = QString::fromStdString("Scale");
    renderText(startX-10,startY+steps+20,text,font);

    //write labels
    float val = 0.0;
    float jump=0;
    int numstep=4;
    for(int i=0;i<=numstep;i++)
    {
        jump = (i/(float)numstep)*steps;
        val = (i/(float)numstep);

        stringstream ss;
        ss<<val;
        QString text = QString::fromStdString(ss.str());
        renderText(startX+15, this->height - (startY+jump),text,font);
    }
}

pcp_openGL_window::pcp_openGL_window(QWidget *parent) : QGLWidget(parent)
{
    this->numVars=3;
    this->height = 250;
    this->width = 780;
    this->currentT = 1;
    this->axisStartX = 120;
    this->axisStartY = 40;
    this->drawHeight = this->height - 1.6*axisStartY;
    this->drawWidth = 400;
    this->axisDiff = this->drawWidth/(this->numVars-1);

    //initialize the sliders
    for(int i=0;i<numVars;i++)
    {
        upperSlider.push_back(vec2(i*axisDiff+axisStartX,axisStartY+drawHeight));
        lowerSlider.push_back(vec2(i*axisDiff+axisStartX,axisStartY));
    }

    this->colorTF = vtkSmartPointer<vtkColorTransferFunction>::New();
    this->colorTF = load_ctf();

    //cmapTable = load_colormap();
}

void pcp_openGL_window :: initializeGL()
{
    glClearColor(1,1,1,1);
}

vtkSmartPointer<vtkColorTransferFunction> pcp_openGL_window :: load_ctf()
{
    vtkSmartPointer<vtkColorTransferFunction> ctable = vtkSmartPointer<vtkColorTransferFunction>::New();
    string line;
    vector <float> cvals;

    //Read
    ifstream readinFis;
    //readinFis.open("../rainbow_ctf.txt");
    readinFis.open("../custom_3.txt");

    while(!readinFis.eof())
    {
        getline(readinFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, ",");

            if(v.size()>0)
            {
                cvals.push_back(v[0]);
            }
        }
    }

    readinFis.close();

    vector<float> datavals;

    for(int i=0;i<cvals.size();i=i+4)
    {
        datavals.push_back(cvals[i]);
    }

    float maxval=datavals[datavals.size()-1];
    float minval=datavals[0];

    for(int i=0;i<datavals.size();i++)
    {
        datavals[i] = (datavals[i]-minval)/(maxval-minval);
    }

    int p=0;
    for(int i=0;i<cvals.size();i=i+4)
    {
        cvals[i] = datavals[p++];
    }

    for(int i=0;i<cvals.size();i=i+4)
    {
        ctable->AddRGBPoint(cvals[i],cvals[i+1],cvals[i+2],cvals[i+3]);
    }

    return ctable;
}

void pcp_openGL_window :: resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h); // set origin to bottom left corner
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void pcp_openGL_window :: mouseMoveEvent(QMouseEvent *event)
{
    //cout<<event->x()<<" "<<this->height-event->y()<<endl;
}

void pcp_openGL_window :: mousePressEvent(QMouseEvent * event)
{
    int press_x = event->x(); int press_y = height-event->y();
    int threshold=10;

    if(event->button() == Qt::LeftButton)
    {
        for(int i=0;i<numVars;i++)
        {
            if(abs(upperSlider[i].x-press_x)<=threshold)
            {
                if(press_y<=axisStartY+drawHeight && press_y>=axisStartY)
                {

                    if(press_y >= lowerSlider[i].y)
                    {
                        upperSlider[i].y = press_y;

                    }
                }
            }

        }
    }
    if(event->button() == Qt::RightButton)
    {
        for(int i=0;i<numVars;i++)
        {
            if(abs(lowerSlider[i].x-press_x)<=threshold)
            {
                if(press_y<=axisStartY+drawHeight && press_y>=axisStartY)
                {
                    if(press_y <= upperSlider[i].y)
                    {
                        lowerSlider[i].y = press_y;
                    }
                }
            }
        }
    }

    updateGL();
}

void pcp_openGL_window :: mouseReleaseEvent(QMouseEvent * event)
{
    //cout<<event->x()<<" "<<this->height-event->y()<<endl;

    //    if(event->button() == Qt::RightButton)
    //    {
    //        //extend here
    //    }
}

void pcp_openGL_window :: paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_LINE_SMOOTH);
    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_LINE_SMOOTH );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //draw colormap
    draw_colormap(700,30);

    minvals[0] = uvelRange.x;
    minvals[1] = entropyRange.x;
    minvals[2] = temperatureRange.x;
    maxvals[0] = uvelRange.y;
    maxvals[1] = entropyRange.y;
    maxvals[2] = temperatureRange.y;
    float scalar;

    QFont font;
    font.setPointSize(10);
    
    //Draw bounding box
    /////////////////////////////////////////////
    glLineWidth(2);
    glColor4f(0.31,0.31,0.31,1.0);

    glBegin(GL_LINES);
    glVertex2f(axisStartX,axisStartY);
    glVertex2f(axisStartX,axisStartY+drawHeight);

    glVertex2f(axisStartX,axisStartY);
    glVertex2f(axisStartX+drawWidth,axisStartY);

    glVertex2f(axisStartX,axisStartY+drawHeight);
    glVertex2f(axisStartX+drawWidth,axisStartY+drawHeight);

    glVertex2f(axisStartX+drawWidth,axisStartY);
    glVertex2f(axisStartX+drawWidth,axisStartY+drawHeight);
    glEnd();

    //draw vertical time step lines
    ////////////////////////////////////////////////
    for(int j=0;j<axisStartX+drawWidth;j=j+this->axisDiff)
    {
        glLineWidth(1);
        glColor4f(0.31,0.31,0.31,0.75);
        glBegin(GL_LINES);
        glVertex2f(axisStartX+j,axisStartY);
        glVertex2f(axisStartX+j,axisStartY+drawHeight);
        glEnd();
    }

    //draw only there are points to draw
    ////////////////////////////////////////////
    if(this->dataInRange.size()>0)
    {
        float first,second;
        for(int i=0;i<this->dataInRange.size();i++)
        {
            int count=0;
            for(int j=0;j<this->dataInRange[i].size()-1;j++)
            {
                //convert to pixel space
                first = axisStartY + drawHeight*this->dataInRange[i][j];
                second = axisStartY + drawHeight*this->dataInRange[i][j+1];

                if(first >= lowerSlider[j].y && first <= upperSlider[j].y && second >= lowerSlider[j+1].y && second <= upperSlider[j+1].y)
                {
                    count++;
                }
            }

            if(count==2) // this means line is valid
            {
                for(int j=0;j<this->dataInRange[i].size()-1;j++)
                {
                    //convert to pixel space
                    first = axisStartY + drawHeight*this->dataInRange[i][j];
                    second = axisStartY + drawHeight*this->dataInRange[i][j+1];

                    //filter based on slider selection
                    if(first >= lowerSlider[j].y && first <= upperSlider[j].y && second >= lowerSlider[j+1].y && second <= upperSlider[j+1].y)
                    {
                        glLineWidth(0.5);
                        glBegin(GL_LINES);

                        scalar = this->dataInRange[i][j]; //(this->dataInRange[i][j]-minvals[j])/(maxvals[j]-minvals[j]);
                        colorval = this->colorTF->GetColor(1-scalar);
                        glColor4f(colorval[0],colorval[1],colorval[2],0.7);

                        glVertex2f((axisStartX+j*axisDiff),first);

                        scalar = this->dataInRange[i][j+1];; //(this->dataInRange[i][j+1]-minvals[j+1])/(maxvals[j+1]-minvals[j+1]);
                        colorval = this->colorTF->GetColor(1-scalar);
                        glColor4f(colorval[0],colorval[1],colorval[2],0.7);

                        glVertex2f((axisStartX+(j+1)*axisDiff),second);
                        glEnd();
                    }
                }
            }

        }
    }

    //write axes names
    font.setPointSize(15);
    glColor4f(0,0,0,1);
    //string varnames[3] = {"Density","Temperature", "Rho"};
    string varnames[3] = {"Pressure","Velocity", "QVapor"};
    int id=0;
    for(int j=0;j<axisStartX+drawWidth;j=j+axisDiff)
    {
        stringstream ss;
        ss<<varnames[id++];
        QString text = QString::fromStdString(ss.str());
        renderText(axisStartX+j-40, this->height-axisStartY+25,text,font);
    }

    //axes range change: process mouse interaction here
    ////////////////////////////////////////////////////
    glPointSize(16);
    glBegin(GL_POINTS);
    glLineWidth(1);
    for(int i=0;i<numVars;i++)
    {
        glColor4f(0,0,1,1);
        glVertex2f(upperSlider[i].x,upperSlider[i].y);
        glVertex2f(lowerSlider[i].x,lowerSlider[i].y);
    }
    glEnd();

    //write slider labels
    ///////////////////////////
    font.setPointSize(12);
    stringstream ss1;
    ss1.precision(3);
    QString text;

    ss1.str("");
    ss1 << this->uvelRange.x + (this->uvelRange.y-this->uvelRange.x)*((upperSlider[0].y-axisStartY)/(drawHeight));
    text = QString::fromStdString(ss1.str());
    renderText(upperSlider[0].x-45, height-upperSlider[0].y ,text,font);

    ss1.str("");
    ss1 << this->entropyRange.x + (this->entropyRange.y-this->entropyRange.x)*((upperSlider[1].y-axisStartY)/(drawHeight));
    text = QString::fromStdString(ss1.str());
    renderText(upperSlider[1].x-45, height-upperSlider[1].y ,text,font);

    ss1.str("");
    ss1 << this->temperatureRange.x + (this->temperatureRange.y-this->temperatureRange.x)*((upperSlider[2].y-axisStartY)/(drawHeight));
    text = QString::fromStdString(ss1.str());
    renderText(upperSlider[2].x-45, height-upperSlider[2].y ,text,font);

    ss1.str("");
    ss1 << this->uvelRange.x + (this->uvelRange.y-this->uvelRange.x)*((lowerSlider[0].y-axisStartY)/(drawHeight));
    text = QString::fromStdString(ss1.str());
    renderText(lowerSlider[0].x-45, height-lowerSlider[0].y ,text,font);

    ss1.str("");
    ss1 << this->entropyRange.x + (this->entropyRange.y-this->entropyRange.x)*((lowerSlider[1].y-axisStartY)/(drawHeight));
    text = QString::fromStdString(ss1.str());
    renderText(lowerSlider[1].x-45, height-lowerSlider[1].y ,text,font);

    ss1.str("");
    ss1 << this->temperatureRange.x + (this->temperatureRange.y-this->temperatureRange.x)*((lowerSlider[2].y-axisStartY)/(drawHeight));
    text = QString::fromStdString(ss1.str());
    renderText(lowerSlider[2].x-45, height-lowerSlider[2].y ,text,font);

    //write axes labels: variable ranges
    ////////////////////////////////////
    int numLabel=3;
    glColor4f(0,0,0,1);
    //uvel
    for(int j=1;j<numLabel;j++)
    {
        stringstream ss;
        ss.precision(3);
        ss << this->uvelRange.x + (this->uvelRange.y-this->uvelRange.x)*(j/(float)numLabel);
        QString text = QString::fromStdString(ss.str());
        renderText(axisStartX - 50, this->height-(axisStartY + j*drawHeight/(float)numLabel) ,text,font);
    }

    //entropy
    for(int j=1;j<numLabel;j++)
    {
        stringstream ss;
        ss.precision(3);
        ss << this->entropyRange.x + (this->entropyRange.y-this->entropyRange.x)*(j/(float)numLabel);
        QString text = QString::fromStdString(ss.str());
        renderText(axisStartX + axisDiff - 45, this->height-(axisStartY + j*drawHeight/(float)numLabel) ,text,font); //assuming axisDiff is 200
    }

    //temperature
    for(int j=1;j<numLabel;j++)
    {
        stringstream ss;
        ss.precision(3);
        ss << this->temperatureRange.x + (this->temperatureRange.y-this->temperatureRange.x)*(j/(float)numLabel);
        QString text = QString::fromStdString(ss.str());
        renderText(axisStartX +drawWidth+10, this->height-(axisStartY + j*drawHeight/(float)numLabel) ,text,font);
    }


    QImage img;
    img = grabFrameBuffer();
    img.save("../pcp_screenshot.jpg");
}
