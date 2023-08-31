#include "GlobalPlanner.h"

// -------------------------------------------------------------
//                         Public methods 
// -------------------------------------------------------------
// Constructor 
// -------------------------------------------------------------
GlobalPlanner::GlobalPlanner(SparseAstar& astar): Node("global_planner")
{
    counter_ = 0;
    // int queue_size = 10;
    
    this->sparse_astar_ = &astar;

    path_pub_ = this->create_publisher<drone_interfaces::msg::Waypoints>(
        "/global_waypoints", 10);
        
    timer_ = this->create_wall_timer(
    std::chrono::milliseconds(1000), std::bind(&GlobalPlanner::publishPath, this));

    agent_pos_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/my_published_msg", 10, std::bind(&GlobalPlanner::topicCallback, 
        this, std::placeholders::_1));
}

// -------------------------------------------------------------
//                         Private methods 
// -------------------------------------------------------------

void GlobalPlanner::agentPosCb(
    const std_msgs::msg::String::SharedPtr msg) const
{
    RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
}

// -------------------------------------------------------------

void GlobalPlanner::publishPath()
{
    std::vector<StateInfo> path = sparse_astar_->searchPath();

    // drone_in
    drone_interfaces::msg::Waypoints msg;

    for (int i=0; i< int(path.size()); i++)
    {
        geometry_msgs::msg::Point wp;
        wp.x = path[i].pos.x;
        wp.y = path[i].pos.y;
        wp.z = path[i].pos.z;
        msg.points.push_back(wp);
        msg.heading.push_back(path[i].psi_dg);

        // RCLCPP_INFO(this->get_logger(), "Path point %d: %f, %f, %f", i, 
        //     wp.x, wp.y, wp.z);
    }
    
    // msg.points = waypoints;

    path_pub_->publish(msg);
    printf("Published path\n");
}

// -------------------------------------------------------------

void GlobalPlanner::topicCallback(
    const std_msgs::msg::String::SharedPtr msg) const
{
    RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
}

// -------------------------------------------------------------

GlobalPlanner::~GlobalPlanner()
{
    RCLCPP_INFO(this->get_logger(), "Destroying GlobalPlanner");
}