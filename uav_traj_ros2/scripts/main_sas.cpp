#include <iostream>
#include <unordered_set>
#include <unordered_map>

#include "Astar.h"
#include "GridMap.h"
#include "PositionVector.h"
// #include <matplot/matplot.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>
// #include "rclcpp/rclcpp.hpp"
#include <chrono>

int main()
{
    printf("SparseAstar finding path\n");

    PositionVector fw_pos(-250, -150, 0);
    PositionVector goal_pos(750,750,25);
    float radius_m = 5.0f;
    float theta_dg = 0.0f;
    float psi_dg = 180.0f;
    float max_psi_turn_dg = 45.0f;
    float max_leg_segment_m = 25;
    FWAgent fw_agent(fw_pos, goal_pos, 
        radius_m, theta_dg, 
        psi_dg, 
        max_psi_turn_dg, max_leg_segment_m);

    fw_agent.setPsi(psi_dg);
    GridMap gridmap;
    gridmap.setGridSize(-500, -500, 0, 1000, 1000, 50);
    fw_agent.setGoalPosition(goal_pos.x, goal_pos.y, goal_pos.z);

    //set seed for random number generator
    srand(2);
    int n_obstacles = 150;
    // insert 20 random obstacles
    for (int i = 0; i< n_obstacles; i++)
    {   
        
        int x = gridmap.getGRID_MIN_X()+ 150  + rand() % \
            (gridmap.getGRID_MAX_X() - 150 - gridmap.getGRID_MIN_X()+ 150);

        int y = gridmap.getGRID_MIN_Y() + 150 + rand() % \
            (gridmap.getGRID_MAX_Y() - 150 - gridmap.getGRID_MIN_Y() + 150);

        int z = gridmap.getGRID_MIN_Z() + 10 + rand() % \
            (gridmap.getGRID_MAX_Z() - 10 - gridmap.getGRID_MIN_Z() + 10);

        int rand_radius = 5 + rand() % 10;

        // make sure obstacle is not too close to agent
        if (sqrt(pow(x - fw_agent.getPosition().x, 2) + 
            pow(y - fw_agent.getPosition().y, 2) + 
            pow(z - fw_agent.getPosition().z, 10)) < 100)
            continue;

        if (sqrt(pow(x - fw_agent.getGoalPosition().x, 2) + 
            pow(y - fw_agent.getGoalPosition().y, 2) + 
            pow(z - fw_agent.getGoalPosition().z, 10)) < 100)
            continue;   

        Obstacle *obstacle = new Obstacle(x, y, z, 
            float(rand_radius), 1);
        
    }

    SparseAstar sparse_astar(gridmap, fw_agent);

    //time the search
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<StateInfo> path = sparse_astar.searchPath();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf("Elapsed time: %f\n", elapsed.count());


    return 0;
}
