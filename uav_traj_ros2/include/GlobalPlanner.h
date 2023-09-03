#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "drone_interfaces/msg/waypoints.hpp"
#include "drone_interfaces/msg/telem.hpp"
#include "drone_interfaces/msg/ctl_traj.hpp"

// #include "mavros_msgs/msg/position_target.hpp"
#include "Astar.h"
#include "GridMap.h"
#include "PositionVector.h"

// Have agent parameters set in Yaml file or config
class GlobalPlanner : public rclcpp::Node
{
    public:
        // Simple constructor to initialize the node with its publishers 
        // and subscribers
        GlobalPlanner(SparseAstar& sparse_astar);

        void publishObstacles();

        ~GlobalPlanner();

        // callback function f

    private:
        
        int counter_;
        SparseAstar* sparse_astar_ = NULL;

        StateInfo agent_info_ = StateInfo(0, 0, 0,0, 0, 0);
        StateInfo traj_info_ = StateInfo(0, 0, 0, -1000, -1000, -1000);

        // drone_in
        drone_interfaces::msg::Waypoints old_path_msg;
        
        rclcpp::TimerBase::SharedPtr timer_, obs_timer_;
        rclcpp::Subscription<drone_interfaces::msg::Telem>::SharedPtr 
            agent_pos_sub_;

        rclcpp::Subscription<drone_interfaces::msg::CtlTraj>::SharedPtr
            agent_traj_sub_;

        // subscribe to agent goal position
        void agentPosCb(
            const drone_interfaces::msg::Telem::SharedPtr msg) ;
        
        // subscribe to agent trajectory
        void agentTrajCb(
            const drone_interfaces::msg::CtlTraj::SharedPtr msg) ;

        void updateAgentInfo();
        void updateTrajInfo();
            
        // subscribe to obstacles information -> future use occupancy grid
        //void obstacles_cb(const std_msgs::msg::String::SharedPtr msg) const;

        // publish path to goal position
        rclcpp::Publisher<drone_interfaces::msg::Waypoints>::SharedPtr path_pub_;  

        rclcpp::Publisher<drone_interfaces::msg::Waypoints>::SharedPtr obs_pub_;  

        //void publish_path(const std::vector<PositionVector>& path) const;
        void publishPath();

        // Checks if there are new obstacles in the grid
        bool anyNewObstacles();

        // Update obstacles information for the grid
        void updateObstacles();

        // Update goal position

        // Update agent position
};