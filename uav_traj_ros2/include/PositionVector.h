#pragma once
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <vector>

class PositionVector
{
    public:
        PositionVector(double x, double y, double z);
        const PositionVector* getPostionVector() {return this;}
        const std::vector<double> getAsVectorDouble() {return {x, y, z};}
        const std::vector<int> getAsVectorInt() {return {int(x), int(y), int(z)};}

        // // comparison to check if two vectors are equal
		inline bool operator== (const PositionVector& v) 
            const { return (x == v.x) && (y == v.y) && (z == v.z); }
		
        inline const PositionVector operator+ (const PositionVector& v) 
            const { return PositionVector(x + v.x, y + v.y, z+v.z); }

        void setX(double x) {this->x = x;}
        void setY(double y) {this->y = y;}
        void setZ(double z) {this->z = z;}

        //returns the change 
		static PositionVector getDelta(const PositionVector& v1, const PositionVector& v2) 
        { return PositionVector(abs(v1.x - v2.x), abs(v1.y - v2.y), abs(v1.z - v2.z));}

        double x;
        double y;
        double z;

};


// Used to represent the state of the agent 
class StateInfo
{   
    public: 

        // This is me being lazy
        PositionVector pos = PositionVector(0, 0, 0);
        double theta_dg;
        double psi_dg;
        double phi_dg;

        StateInfo(double x, double y, double z, 
            double phi_dg, double theta_dg, double psi_dg);
        
        StateInfo(PositionVector& pos, 
            double phi_dg, double theta_dg, double psi_dg);
    
        const PositionVector getPostionVector() {return pos;}
        const double getThetaDg() {return theta_dg;}
        const double getPsiDg() {return psi_dg;}

        //set values of state
        void setState(double x, double y, double z, 
            double theta_dg, double psi_dg, double phi_dg);

        // void setState(PositionVector& pos, double theta_dg, double psi_dg);

};
