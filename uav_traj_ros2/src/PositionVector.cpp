#include <PositionVector.h>

PositionVector::PositionVector(double x, double y, double z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}


StateInfo::StateInfo(double x, double y, double z, 
    double phi_dg, double theta_dg, double psi_dg)
{
    this->pos = PositionVector(x, y, z);
    this->phi_dg = phi_dg;
    this->theta_dg = theta_dg;
    this->psi_dg = psi_dg;
}

StateInfo::StateInfo(PositionVector& pos,
    double phi_dg, double theta_dg, double psi_dg)
{
    this->pos = pos;
    this->phi_dg = phi_dg;
    this->theta_dg = theta_dg;
    this->psi_dg = psi_dg;
}

void StateInfo::setState(double x, double y, double z, 
    double theta_dg, double psi_dg)
{
    this->pos = PositionVector(x, y, z);
    this->theta_dg = theta_dg;
    this->psi_dg = psi_dg;
}

