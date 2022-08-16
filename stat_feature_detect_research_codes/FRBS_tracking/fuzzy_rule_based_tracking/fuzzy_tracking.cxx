#include <cpp_headers.h>
#include <rule_base_func.h>
#include <glm_headers.h>
#include <feature_vector.h>
#include <util_funcs.h>
#include <python2.7/Python.h>

PyObject *pName, *pModule, *pDict, *pFunc;
PyObject *pArgs, *pValue;

using namespace std;
using namespace cv;

ofstream logfile;
ofstream graphfile;
ofstream conffile;

//////////////////////////////////////////////////////
// Measure time function
///////////////////////////////////////////////////////

double clkbegin, clkend;
double t=0;
double python_overhead_start,python_overhead_end,pca_split_end,pca_split_start;
/////////////////////////////////////////////////////////////////////////////////

int python_setup_function(char* py_file, char* func_name, int val)
{
    cout<<"Split check initiated at time step: "<<val<<endl;
    int matched_splitted_id = 0;

    pName = PyString_FromString(py_file);
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL)
    {
        pFunc = PyObject_GetAttrString(pModule, func_name);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc))
        {
            //allocate memory for parameter conversion from c to python and convert
            pArgs = PyTuple_New(1);
            pValue = PyInt_FromLong(val);
            PyTuple_SetItem(pArgs, 0, pValue);

            pValue = PyObject_CallObject(pFunc, pArgs); // call python function with parameter
            Py_DECREF(pArgs);

            if (pValue != NULL)
            {
                matched_splitted_id = PyInt_AsLong(pValue);

                if(matched_splitted_id==-1)
                    cout<<"Some trouble in python function"<<endl;

                Py_DECREF(pValue);
            }
            else
            {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                cout<<"Call failed"<<endl;
            }
        }

        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else
    {
        PyErr_Print();
        cout<<"Failed to load: "<<py_file<<endl;
    }

    return matched_splitted_id;
}

int main(int argc, char** argv)
{
    ///////////////////////////////////////////////////////////////////////////////
    // python env path set
    ///////////////////////////////////////////////////////////////////////////////
    setenv("PYTHONPATH","/home/soumya/Dropbox/Codes/fuzzy_rule_based_tracking/",1);
    Py_Initialize();

    // to turn off vtk warning messages
    vtkObject::GlobalWarningDisplayOff();
    conffile.open("conf_val_file.txt");
    logfile.open("logfile.txt");
    graphfile.open("graphfile.txt");
    const int rule_num = 3; //TODO
    const int input_dim_num = 4; //TODO
    int timeSteps=18;//TODO
    int initStep=10; //TODO
    float threshold = 7.0; //Isabel = -1.0, 7 for vortex, -0.001 for tornado // TODO
    float conf_level = 0.70;
    int num_segments=0;
    float mass_threshold = 0.25;
    vector<Feature_class> init_feature_list;
    float xyzf[3];
    int dim[3];
    vector< vector<float> > graph_data;
    int isSplit=0;
    stringstream tt;
    tt<<initStep;
    string inputFilename,path;
    int splitid1,splitid2;
    float timetime=0;

    ////////////////////////////////////////////////////////////////////////
    ////Vortex data
    ////////////////////////////////////////////////////////////////////////
    path = "/home/soumya/Test_DataSet/vortex/vti/";
    inputFilename = path + "vortex_" + tt.str() + ".vti";

    //    //f1: for testing vortex t = 1 ~ 20 (works fine for all time steps): mind = 0.25, split t = 14 3D MATCHING WORKS, PCA works,(0,1) , conf = 0.7, mass = 0.25, works fine
    //    xyzf[0] = 63;
    //    xyzf[1] = 55;
    //    xyzf[2] = 2.5;

       // //f2: for testing vortex t = 1 ~ 15 : mind = 0.25, split t = 11 3D MATCHING WORKS, PCA works (11,14) wrong split detection at t = 16 conf = 0.7, mass = 0.25 wroks fine
       // xyzf[0] = 17.5;
       // xyzf[1] = 67.5;
       // xyzf[2] = 85.0;

    // //f3: for testing vortex t = 1 ~ 13 : mind = 0.20, split t = 11 3D MATCHING WORKS, (18,19) conf = 0.7, mass = 0.25
    // xyzf[0] = 62.5;
    // xyzf[1] = 60;
    // xyzf[2] = 120;

       //f4: for testing vortex t = 10 ~ 30 : mind = 0.25, split t = 25 3D MATCHING WORKS, PCA works (6,8):  merge is at t = 2, works with conf = 0.7
       xyzf[0] = 65;
       xyzf[1] = 100;
       xyzf[2] = 100;

    //    //f5: for testing vortex t = 1 ~ 20 : mind = 0.25 just continuous feature
    //    xyzf[0] = 50;
    //    xyzf[1] = 50;
    //    xyzf[2] = 50;

    //    //f6: for testing vortex t = 1 ~ 20 : mind = 0.25, dies at 19
    //    xyzf[0] = 100;
    //    xyzf[1] = 10;
    //    xyzf[2] = 50;

    /////////////////////////////////////////////////////////////////////
    ////Combustion data
    /////////////////////////////////////////////////////////////////////

       // path = "/home/soumya/Test_DataSet/Combustion/vorticity_vti/";
       // inputFilename = path + "combustion_vorticity_" + tt.str() + ".vti";

    //        //f1: for combustion t = 50 ~ 65 : mind = 0.08 split at 62, (20,21), minpts = 75, 3D MATCHING WORKS with projectedvolume(), PCA works, conf = 0.75, mass = 0.2 works fine
    //        xyzf[0] = 60;
    //        xyzf[1] = 135;
    //        xyzf[2] = 42.5;

       // //f2: for combustion : t = 65 ~ 85 : mind = 0.08 split at 82, (5,6) minpts = 75, 3D MATCHING WORKS with projectedvolume(), PCA works, conf = 0.75, mass = 0.2 works fine
       // xyzf[0] = 187;
       // xyzf[1] = 235;
       // xyzf[2] = 19;

    //    //f3: for combustion : t = 45 ~ 65 : mind = 0.08 split at 55, (14,23), minpts = 75 3D MATCHING WORKS with projectedvolume(), PCA works, conf = 0.75, mass = 0.2 works fine
    //    xyzf[0] = 60;
    //    xyzf[1] = 225;
    //    xyzf[2] = 15;

    /////////////////////////////////////////////////////////////////////
    ////Isabel data: no split, can skip time steps, no split
    /////////////////////////////////////////////////////////////////////

    //    path = "/home/soumya/Test_DataSet/Isabel_vortex/vti/";
    //    inputFilename = path + "isabel_lambda2_" + tt.str() + ".vti";

    //    //for Isabel
    //    xyzf[0] = 142;
    //    xyzf[1] = 60;
    //    xyzf[2] = 7.5;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int index=0; //feature index
    system("rm *.vti");
    system("rm *.vtu");
    system("rm *.vtp");
    system("rm /home/soumya/comb_features/matched/*.csv");
    //system("rm /home/soumya/comb_features/*.csv");

    init_feature_list = locate_target_feature(inputFilename,&index,threshold,xyzf,dim,initStep);

    //Create the target feature with properties
    Feature_class fstar;
    fstar = init_feature_list[index]; //index determines the best match i.e. the target at t=1
    cout<<"initial target feature center: "<<fstar.cog[0]<<" "<<fstar.cog[1]<<" "<<fstar.cog[2]<<endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Processing initial time step ends here
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Initialize some data structures which will be needed for fuzzy based tracking
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    float temp_rule_matrix1[rule_num][input_dim_num];
    float temp_rule_matrix2[rule_num][input_dim_num];
    string line;
    vector<glm::vec2> temp_rule_matrix;

    //Read the parameters of the trained fuzzy rule based system
    ///////////////////////////////////////////////////////////////////
    ifstream readoutFis;
    readoutFis.open("../outputmfs.txt");

    //Read outmfs
    getline(readoutFis, line);
    vector<float> outmfs = split(line, ",");

    //Read inmfs
    ifstream readinFis;
    readinFis.open("../inputmfs.txt");

    while(!readinFis.eof())
    {
        getline(readinFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, ",");

            if(v.size()>0)
            {
                temp_rule_matrix.push_back(glm::vec2(v[0],v[1]));
            }
        }
    }

    int ij=0;
    for(int qq=0;qq<input_dim_num;qq++)
    {
        for(int jj=0;jj<rule_num;jj++)
        {
            glm::vec2 v21 = temp_rule_matrix[ij++];
            temp_rule_matrix1[jj][qq] = v21.x; // sigma vals
            temp_rule_matrix2[jj][qq] = v21.y; // mean vals
        }
    }

    temp_rule_matrix.clear();

    ///////////////////////////////////////////////
    //Fuzzy rule based inference system creation
    ///////////////////////////////////////////////
    Rule_Based_System rulebase;
    rulebase.num_rules = rule_num;
    rulebase.num_input_dim = input_dim_num;
    rulebase.fuzzy_system_type = "tsk";

    for(int qq=0;qq<rule_num;qq++)
    {
        Rules rule;
        rule.membership_func_type = "gmf";

        for(int jj=0;jj<input_dim_num;jj++)
        {
            Membership_func mm;
            mm.sigma = temp_rule_matrix1[qq][jj]; // sigma vals
            mm.mean = temp_rule_matrix2[qq][jj]; // mean vals
            rule.inputmfs.push_back(mm);
        }

        for(int jj=0;jj<=input_dim_num;jj++)
            rule.out_params.push_back(outmfs[jj]);

        rulebase.rules.push_back(rule);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //Iterate over each time step and track the selected feature finally!!
    ////////////////////////////////////////////////////////////////////////////////////////////////
    for(int ii=initStep+1; ii<initStep+timeSteps;ii++)
    {
        system("rm /home/soumya/comb_features/*.csv");
        isSplit=0;

        //This is the max possible distance a feature can move inside the data domain
        float maxdist = sqrt(dim[0]*dim[0] + dim[1]*dim[1] + dim[2]*dim[2]);
        vector<Feature_class> current_feature_list;
        vector<float> distance_cog;
        vector<float> distance_mass;
        vector<float> distance_blob;
        vector<float> distance_cbbox;
        map<float,Feature_vector> feature_prop;
        vector<glm::vec2> evaluated_vals;
        int split_id;
        int match = 0;
        stringstream pp;
        pp<<ii;
        vector<float> graph_one_time;
        map<int,glm::vec2> feature_comb_map;

        /////////////////////////////////////////////////////////////////////////////////////////////////
        //Generate input file name for selected data set
        /////////////////////////////////////////////////////////////////////////////////////////////////

        //inputFilename = path + "tornado_lambda2_" + pp.str() + ".vti";
        inputFilename = path + "vortex_" + pp.str() + ".vti";
        //inputFilename = path + "isabel_lambda2_" + pp.str() + ".vti";
        //inputFilename = path + "combustion_vorticity_" + pp.str() + ".vti";
        /////////////////////////////////////////////////////////////////////////////////////////////////

        //Read data
        vtkSmartPointer<vtkImageData> imageData1 = read_data(inputFilename);

        ////////////////////////////////////////////////////
        // start time measure: excluding data loading
        ///////////////////////////////////////////////////
        clkbegin = rtclock();

        //Segments and returns the unstructured grid version of data
        vtkSmartPointer<vtkUnstructuredGrid> ug1 = segment_feature_from_data(&num_segments,threshold,imageData1,ii);

        //Reads the segmented unstructured grid data in and creates the feature list
        construct_feature_list(ug1,&current_feature_list,num_segments,ii);

        //Computes the attributes for each feature for correspondence solving and tracking
        // input: feature list
        // output: a map datastructure containing feature comparison attributes
        compute_feature_attribute_variations(current_feature_list,&feature_prop,fstar,&distance_cog,&distance_mass,&distance_blob,&distance_cbbox,maxdist);

        // Find feature correspendence using fuzzy rule based system
        map<float,Feature_vector>::iterator it = feature_prop.begin();
        for(it=feature_prop.begin(); it!=feature_prop.end(); ++it)
        {
            glm::vec2 rett = evaluate_rulebase(rulebase,it->second);
            evaluated_vals.push_back(rett);
        }

        //generate log outputs
        for(int qq=0;qq<evaluated_vals.size();qq++)
        {
            logfile<<"time step: "<<ii<<" score: "<<evaluated_vals[qq].x<<" id: "<<evaluated_vals[qq].y<<endl;
            graph_one_time.push_back(evaluated_vals[qq].x);
        }

        graph_data.push_back(graph_one_time);

        //Finally get the matched feature index and assign it to match
        /////////////////////////////////////////////////////////////////
        glm::vec2 return_value = find_matched_index(evaluated_vals,ii);
        match = (int)return_value.y;
        ///////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////
        // Do PCA here for matched feature
        ////////////////////////////////////////////////////////////////////////////////////////////
        double p[3];
        vtkDataArray *matched_point_ary = current_feature_list[match].surface->GetPoints()->GetData();
        Mat compressed;
        Mat time_line_mat(matched_point_ary->GetNumberOfTuples(),3,CV_32FC1);
        //Populate data for PCA
        for(int k=0;k<matched_point_ary->GetNumberOfTuples();k++)
        {
            matched_point_ary->GetTuple(k,p);
            time_line_mat.at<float>(k,0) = p[0];
            time_line_mat.at<float>(k,1) = p[1];
            time_line_mat.at<float>(k,2) = p[2];
        }

        //Do PCA: compressed matrix will have the required values
        PCA_analysis(time_line_mat, 3, time_line_mat, compressed);

        ///////////////////////////////////////////////////////////////////////
        // Write the csv file of matched feature
        ///////////////////////////////////////////////////////////////////////
        string comb_file = "/home/soumya/comb_features/matched/matched_feature_" + pp.str() + ".csv";
        ofstream  comb_file_ptr;
        comb_file_ptr.open(comb_file.c_str());
        comb_file_ptr<<"x"<<","<<"y"<<","<<"z"<<endl;

        for(int k=0;k<compressed.rows;k++)
            comb_file_ptr<<round(compressed.at<float>(k,0))<<","<<round(compressed.at<float>(k,1))<<","<<round(compressed.at<float>(k,2))<<endl;

        comb_file_ptr.close();

        ///////////////////////////////////////////////////////////////////////////
        //New overlap and PCA based split check section
        ///////////////////////////////////////////////////////////////////////////
        if(return_value.x < conf_level && distance_mass[match] > mass_threshold)
        {
            ///////////////////////////////////////////////////////////////////////////////////////////////
            // add the detected feature itself into comparison
            string ff_file = "/home/soumya/comb_features/matched/matched_feature_" + pp.str() + ".csv";
            string copy_file = "cp " + ff_file + " /home/soumya/comb_features/temp.csv";
            system(copy_file.c_str());

            //writes the csv files for python PCA interface
            pca_split_start = rtclock();
            feature_comb_map = create_feature_combination(current_feature_list,fstar,maxdist,ii,match);
            pca_split_end = rtclock();
            cout<<(pca_split_end-pca_split_start)<<" "<<"secs PCA based split check overhead at time step: "<<ii<<endl;

            //compute Jaccard index and get best match for spllited feature
            python_overhead_start = rtclock();
            split_id = python_setup_function("py_function","main_func",ii);
            python_overhead_end = rtclock();
            cout<<(python_overhead_end-python_overhead_start)<<" "<<"secs python overhead at time step: "<<ii<<endl;
        }

        conffile<<return_value.x<<endl;

        /////////////////////print the map here////////////////////////////////////////////////////////////
        map<int,glm::vec2>::iterator it1 = feature_comb_map.begin();
        for (it1=feature_comb_map.begin(); it1!=feature_comb_map.end(); ++it1)
        {
            if(split_id == it1->first)
            {
                cout << "split detected and matched combination is: "<<it1->first << " => " << it1->second.x << ","<<it1->second.y<<endl;
                splitid1 = it1->second.x;
                splitid2 = it1->second.y;
                isSplit=1;
            }
        }

        if(!isSplit)
            cout<<"no split happened "<<endl;

        ////////////////////////////////////////////////
        //Reassign current feature to matched feature
        ////////////////////////////////////////////////
        fstar = current_feature_list[match];

        //write tracked resulting volumes out for visualization
        ///////////////////////////////////////////////////////////////////////////////
        if(!isSplit)
            generate_output_volume(ii,imageData1,current_feature_list,ug1,match,threshold);
        else if(isSplit)
            generate_output_volume_split(ii,imageData1,current_feature_list,ug1,splitid1,splitid2,threshold);

        cout<<"Matched feature at time: "<<ii<< " is: "<<match<< " distance changed: "<<distance_cog[match]<< " mass changed: "
           <<distance_mass[match]<<" cog changed: "<<distance_cbbox[match]<<" "<<" FIS val: :"<<return_value.x<<endl;

        logfile <<"Matched feature at time: "<<ii<< " is: "<<match<< " distance changed: "<<distance_cog[match]<< " mass changed: "
               <<distance_mass[match]<<" cog changed: "<<distance_cbbox[match]<<" "<<" FIS val: :"<<return_value.x<<endl<<endl;

        ///////////////////////////////////////////////////////
        // test on training data
        /*Feature_vector train_feature;
        train_feature.feature_vec.push_back(0.1000702225);
        train_feature.feature_vec.push_back(0.5017716);
        train_feature.feature_vec.push_back(0.500101819);
        train_feature.feature_vec.push_back(0.100074799);
        train_feature.id = 999;
        glm::vec2 rett_test = evaluate_rulebase(rulebase,train_feature);
        cout<<"res on test data prediction: "<<rett_test.x<<endl;*/

        clkend = rtclock();
        t = clkend - clkbegin;
        timetime += t;
        cout<<t<<" "<<"secs took to process time step: "<<ii<<endl<<endl;
    }

    cout<<"average running time per timestep: "<<timetime/(timeSteps-1)<<endl;


    ///generate graph file
    /*for(int ii=0; ii<graph_data.size();ii++)
    {
        for(int i=0;i<graph_data[ii].size();i++)
        {
            if(i!=graph_data[ii].size()-1)
                graphfile<<graph_data[ii][i]<<",";
            else
                graphfile<<graph_data[ii][i];
        }

        graphfile<<endl;
    }
    graphfile.close();*/

    logfile.close();
    Py_Finalize();
    conffile.close();

    return 0;
}
