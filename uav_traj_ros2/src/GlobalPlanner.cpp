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

    path_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/my_published_msg", 10);
        
    timer_ = this->create_wall_timer(
    std::chrono::milliseconds(500), std::bind(&GlobalPlanner::publishPath, this));

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
    std::vector<PositionVector> path = sparse_astar_->searchPath();

    for (int i=0; i<path.size(); i++)
    {
        RCLCPP_INFO(this->get_logger(), "Path point %d: %f, %f, %f", i, 
            path[i].x, path[i].y, path[i].z);
    }

    std_msgs::msg::String message;
    message.data = "HELLO WORLD number " + std::to_string(counter_++);
    path_pub_->publish(message);
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