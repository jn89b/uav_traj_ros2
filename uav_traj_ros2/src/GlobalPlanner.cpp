#include "GlobalPlanner.h"

// -------------------------------------------------------------
//                         Public methods 
// -------------------------------------------------------------
// Constructor 
// -------------------------------------------------------------
GlobalPlanner::GlobalPlanner(SparseAstar& sparse_astar_): Node("global_planner")
{
    counter_ = 0;

    this->sparse_astar_ = &sparse_astar_;

    path_pub_ = this->create_publisher<drone_interfaces::msg::Waypoints>(
        "/global_waypoints", 10);
        
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000), 
        std::bind(&GlobalPlanner::publishPath, 
        this));

    agent_pos_sub_ = this->create_subscription<drone_interfaces::msg::Telem>(
        "/enu_telem", 10, 
        std::bind(&GlobalPlanner::agentPosCb, 
        this, std::placeholders::_1));

    agent_traj_sub_ = this->create_subscription<drone_interfaces::msg::CtlTraj>(
        "/trajectory", 10, 
        std::bind(&GlobalPlanner::agentTrajCb, 
        this, std::placeholders::_1));

    // for now only publish obstacles once
    obs_pub_ = this->create_publisher<drone_interfaces::msg::Waypoints>(
        "/obs_positions", 10);

    // publishPath();  
    publishObstacles();

    obs_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10000), 
        std::bind(&GlobalPlanner::publishObstacles, 
        this));
}

// -------------------------------------------------------------
//                         Private methods 
// -------------------------------------------------------------

void GlobalPlanner::agentPosCb(
    const drone_interfaces::msg::Telem::SharedPtr msg) 
{   
    //
    double x = round(msg->x); 
    double y = round(msg->y);
    double z = round(msg->z);

    double psi_dg = msg->yaw * 180.0f / M_PI;
    double theta_dg = msg->pitch * 180.0f / M_PI;
    double phi_dg = msg->roll * 180.0f / M_PI;
        
    // //replace agent_info_ with new agent info
    // StateInfo new_agent_info = StateInfo(pos, theta_dg, psi_dg);
    agent_info_.setState(x, y, z, theta_dg, psi_dg, phi_dg);

}

// -------------------------------------------------------------

void GlobalPlanner::agentTrajCb(
    const drone_interfaces::msg::CtlTraj::SharedPtr msg) 
{
    // get last value to set as start set
    int idx = 4;
    float x = round(msg->y[idx]);
    float y = round(msg->x[idx]); 
    float z = -round(msg->z[idx]);

    float psi_dg = msg->yaw[idx] * 180.0f / M_PI;
    float theta_dg = msg->pitch[idx] * 180.0f / M_PI;
    float phi_dg = msg->roll[idx] * 180.0f / M_PI;
    // update traj info
    // StateInfo traj_info_ = StateInfo(pos, theta_dg, psi_dg);
    traj_info_.setState(x,y,z, theta_dg, psi_dg, phi_dg);

}

// -------------------------------------------------------------

void GlobalPlanner::publishPath()
{
    //callback 
    // rclcpp::spin_some(this->get_node_base_interface());
    printf("Agent position: %f, %f, %f\n", 
     sparse_astar_->getAgent()->getPosition().x,
     sparse_astar_->getAgent()->getPosition().y,
     sparse_astar_->getAgent()->getPosition().z);

    if (traj_info_.psi_dg == -1000)
    {
        printf("No trajectory info\n");
        sparse_astar_->updateAgentPosition(agent_info_.pos, 
            agent_info_.theta_dg, agent_info_.psi_dg);

    }
    else{
        printf("traj_info_ position: %f, %f, %f\n", 
         traj_info_.pos.x,
         traj_info_.pos.y,
         traj_info_.pos.z);
        sparse_astar_->updateAgentPosition(traj_info_.pos, 
            traj_info_.theta_dg, traj_info_.psi_dg);
    }

    std::vector<StateInfo> path = sparse_astar_->searchPath();

    if (path.size() == 0)
    {
        printf("No path found using old path msg\n");
        path_pub_->publish(old_path_msg);
        return;
    }
    else
    {
        drone_interfaces::msg::Waypoints msg;
        for (int i=0; i< int(path.size()); i++)
        {
            geometry_msgs::msg::Point wp;
            wp.x = path[i].pos.x;
            wp.y = path[i].pos.y;
            wp.z = path[i].pos.z;
            msg.points.push_back(wp);
            // rotate by 90 degrees and wrap to 360 
            double psi_wrap = path[i].psi_dg + 90 % 360;
            msg.heading.push_back(psi_wrap);
            msg.roll.push_back(path[i].phi_dg);
            msg.pitch.push_back(path[i].theta_dg);
        }
        
        path_pub_->publish(msg);
        printf("Published path\n");
        old_path_msg = msg;

        return; 
    }
}

// -------------------------------------------------------------

void GlobalPlanner::publishObstacles()
{
    // drone_in
    drone_interfaces::msg::Waypoints msg;

    const std::vector<Obstacle*> obs_list = sparse_astar_->
            getGridMap()->getObstacles(); 

    for (int i=0; i< int(obs_list.size()); i++)
    {
        geometry_msgs::msg::Point wp;

        wp.x = obs_list[i]->getX();
        wp.y = obs_list[i]->getY();
        wp.z = obs_list[i]->getZ();
        msg.points.push_back(wp);
        msg.heading.push_back(obs_list[i]->getRadius());
        msg.pitch.push_back(0);
    }
    
    obs_pub_->publish(msg);
    printf("size of obs list: %d\n", int(obs_list.size()));
    printf("Published obstacles\n");
}


// void GlobalPlanner::topicCallback(
//     const std_msgs::msg::String::SharedPtr msg) const
// {
//     RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
// }

// -------------------------------------------------------------

GlobalPlanner::~GlobalPlanner()
{
    RCLCPP_INFO(this->get_logger(), "Destroying GlobalPlanner");
}