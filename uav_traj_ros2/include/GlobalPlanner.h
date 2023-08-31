#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "drone_interfaces/msg/waypoints.hpp"

#include "Astar.h"
#include "GridMap.h"

// Have agent parameters set in Yaml file or config


class GlobalPlanner : public rclcpp::Node
{
    public:
        // Simple constructor to initialize the node with its publishers 
        // and subscribers
        GlobalPlanner(SparseAstar& astar);
        ~GlobalPlanner();

        // callback function f

    private:
        
        int counter_;
        SparseAstar* sparse_astar_ = NULL;
        
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr agent_pos_sub_;
        
        // subscribe to agent goal position
        void agentPosCb(const std_msgs::msg::String::SharedPtr msg) const;
        void topicCallback(const std_msgs::msg::String::SharedPtr msg) const;

        // subscribe to obstacles information
        //void obstacles_cb(const std_msgs::msg::String::SharedPtr msg) const;

        // publish path to goal position
        rclcpp::Publisher<drone_interfaces::msg::Waypoints>::SharedPtr path_pub_;  
        //void publish_path(const std::vector<PositionVector>& path) const;
        void publishPath();

        // Checks if there are new obstacles in the grid
        bool anyNewObstacles();

        // Update obstacles information for the grid
        void updateObstacles();

        // Update goal position

        // Update agent position
};