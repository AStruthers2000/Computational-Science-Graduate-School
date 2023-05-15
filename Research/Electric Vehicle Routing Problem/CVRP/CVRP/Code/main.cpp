/**
 * @brief Vehicle Routing Problem
 * @author doc. MSc. Donald Davendra Ph.D.
 * @date 3.10.2013
 *
 */

#include <iostream>
#include "VRP.h"

using namespace std;

void TotalDistance(cVRP *VRP);

//! the main function.
/*!
 \return 0 for successful completion
 */
int main ()
{
    //! Initialization of the VRP class
    cVRP* VRP = new cVRP();
    
    //! Calculate a total flow
    TotalDistance(VRP);

    return 0;
}

//! Function to calculate a simple tour.
/*!
 \return no return value
 */
void TotalDistance(cVRP *VRP){
    //! Initilaize a tour
    int *Tour = new int[VRP->GetLocations()];
    
    //! Fill the tour sequentially
    for (int i = 0; i < VRP->GetLocations(); i++) {
        Tour[i] = i+1;
    }
    
    cout << "The total distance traveled is: " << VRP->TourCost(Tour) << endl;
    
    //! Delete the schdule.
    delete [] Tour;

}
