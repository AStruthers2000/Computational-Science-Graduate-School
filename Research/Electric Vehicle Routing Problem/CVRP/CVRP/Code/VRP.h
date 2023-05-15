/**
 * @brief Vehicle Routing Problem
 * @author doc. MSc. Donald Davendra Ph.D.
 * @date 3.10.2013
 *
 */

#ifndef __VRP_H__
#define __VRP_H__

class cVRP{
	
public:
	cVRP();
    
	~cVRP();
    
	float TourCost(int*);
    
    int GetLocations();
    
    float NodeLength(int, int);
    
    float NodeViaDepot(int, int);
	
private:
	float* m_X;
    float* m_Y;
    float* m_Demand;
    float m_Capacity;
    int m_Locations;
    float XD;
    float YD;
};

#endif
