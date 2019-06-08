//Adapted from https://fiftyexamples.readthedocs.io/en/latest/gravity.html
//
#include <string>
#include <math.h>
#include <iostream>
#include <tuple>

using namespace std;

#define G 6.67428e-11

// Assumed scale: 100 pixels = 1AU.
#define AU  (149.6e6 * 1000)     // 149.6 million km, in meters.
#define SCALE  (250 / AU)



class Body
{
    // Access specifier
    public:

    // Data Members
    string name;

    // name = 'Body'
    float mass,vx,vy,px,py;
    
    std::tuple<float, float> attraction(Body other)
    {
      // Compute the distance of the other body.
      float tx = this->px;
      float ty = this->py;
      float ox = other.px;
      float oy = other.py;

      float dx = (ox-tx);
      float dy = (oy-ty);
      float d = sqrt(dx*dx + dy*dy);
      float f = G * this->mass * other.mass / (d*d);

      float theta = atan2(dy, dx);
      float fx = cos(theta) * f;
      float  fy = sin(theta) * f;
      cout << "\nDistance is : " << d;
      return std::make_tuple(fx,fy);
    }
};

void update_info(long step, Body body)
{
    cout << "Step #{}" << step;
    cout << body.name, body.px/AU, body.py/AU, body.vx, body.vy;
    cout << "\n";
}

int main() {

    // Declare an object of class geeks
    Body earth,sun,venus;

    sun.name = "Sun";
    sun.mass = 1.98892 * (10^30);
    sun.px = 0;
    sun.py = 0;
    sun.vx = 0;
    sun.vy = 0;            // 29.783 km/sec

    // accessing data member
    earth.name = "Earth";
    earth.mass = 5.9742 * (10^24);
    earth.px = -1*AU;
    earth.py = 0;
    earth.vx = 0;
    earth.vy = 29.783*1000;            // 29.783 km/sec

    venus.name = "Venus";
    venus.mass = 4.8685 * (10^24);
    venus.px = 0.723 * AU;
    venus.py = 0;
    venus.vy = -35.02 * 1000;
    venus.vx = 0;


    // accessing member function
    auto [fx, fy] = earth.attraction(venus);
    cout << "\nFx is : " << fx << " and Fy is :" << fy;
    //venus.printname();
    //sun.printname();
    return 0;
}

