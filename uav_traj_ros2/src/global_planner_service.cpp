#include "rclcpp/rclcpp.hpp"
#include "srv/capital_full_name.hpp"   // custom service from another package
#include <boost/algorithm/string.hpp>   // boost library to handle string
#include "Astar.h"
#include "GridMap.h"
#include "PositionVector.h"


/*
    Service takes in:
        Agent goal position
        Agent radius
        Agent current position
        
    While running:
        The service will subscribe to the current position of the agent
        The service will publish the path to the goal position
        
    Service publishes:
        Path trajectories to goal

*/


// method to handle the client request and give back a response
// the service gets the name and surname and responses with a capitalized full name
void ComposeFullName(const std::shared_ptr<custom_msg_and_srv::srv::CapitalFullName::Request> request,
        std::shared_ptr<custom_msg_and_srv::srv::CapitalFullName::Response> response)
{
  std::string fullName = request->name + " " + request->surname;    // creation of the full name
  std::string capitalFullName = boost::to_upper_copy<std::string>(fullName);    // capitalize the full name
  response->capitalfullname = capitalFullName;    
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Incoming name: %s" "\tIncoming surname: %s",
              request->name.c_str(), request->surname.c_str());
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sending back the capitalize full name: [%s]", response->capitalfullname.c_str());
}


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("service_node");    // "auto" is "std::shared_ptr<rclcpp::Node>"

  // declaration of the service called "create_cap_full_name", it waits a request from the client
  // that will be handled by the ComposeFullName function
  rclcpp::Service<custom_msg_and_srv::srv::CapitalFullName>::SharedPtr service = 
    node->create_service<custom_msg_and_srv::srv::CapitalFullName>("create_cap_full_name",  &ComposeFullName);
  RCLCPP_INFO(node->get_logger(), "I am ready to capitalize your full name");
  rclcpp::spin(node);   // the service starts to wait and manage requests
  rclcpp::shutdown();
}