/**
 * @brief Vehicle Routing Problem
 * @author doc. MSc. Donald Davendra Ph.D.
 * @date 3.10.2013
 *
 */


/*! \file VRP.h
 \brief A VRP header file.
 */
#include "VRP.h"

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

cVRP::cVRP(){
    float best, temp;
    
    ifstream infile;
    char fname[120]="tai75a.dat";
    infile.open(fname);
    
    if(!infile.is_open()) {
        cout << "Error Opening File.\n";
        exit(1);
    }
    else {
        infile >> m_Locations >> best >> m_Capacity;
        
        infile >> XD >> YD;
        
        m_X = new float[m_Locations];
        m_Y = new float[m_Locations];
        m_Demand = new float[m_Locations];
        
        for (int i = 0; i < m_Locations; i++){
            infile >> temp >> m_X[i] >> m_Y[i] >> m_Demand[i];
        }
    }
    infile.close();
}

cVRP::~cVRP(){
    delete [] m_X;
    delete [] m_Y;
    delete [] m_Demand;
}

int cVRP::GetLocations(){
    
    return m_Locations;
}

float cVRP::NodeLength(int S, int D){
    float distance = 0;
    
    if (S == 0) {
        distance = sqrt(pow(m_X[D-1] - XD,2)+pow(m_Y[D-1] - YD,2));
    }
    else if(D == 0){
        distance = sqrt(pow(XD - m_X[S-1],2)+pow(YD - m_Y[S-1],2));
    }
    else{
        distance = sqrt(pow(m_X[D-1] - m_X[S-1],2)+pow(m_Y[D-1] - m_Y[S-1],2));
    }
    return distance;
}

float cVRP::NodeViaDepot(int S, int D){
    float distance = 0;
    
    distance = sqrt(pow(XD - m_X[S-1],2)+pow(YD - m_Y[S-1],2));
    
    distance += sqrt(pow(m_X[D-1] - XD,2)+pow(m_Y[D-1] - YD,2));
    
    return distance;
}

float cVRP::TourCost(int *Tour){

    float capacity = 0, distance = 0;
    int subtours = 1;
    
    // start the tour from depot to first client
    distance += NodeLength(0, Tour[0]);
    capacity += m_Demand[Tour[0]-1];
    
    // sequentially visit all the clients
    for (int i = 1; i < m_Locations; i++) {
        // If there is still capacity to service another client.
        if(capacity + m_Demand[Tour[i]-1] <= m_Capacity){
            distance += NodeLength(Tour[i-1], Tour[i]);
            capacity += m_Demand[Tour[i]-1];
        }
        else{
            // return to deport and empty the truck
            capacity = m_Demand[Tour[i]-1];
            distance += NodeViaDepot(Tour[i-1], Tour[i]);
            subtours++;
        }
    }
    // return to depot
    distance += NodeLength(Tour[m_Locations-1], 0);
    
    //cout << "the total number of subtours is: " << subtours << endl;
    
    return distance;
    
}