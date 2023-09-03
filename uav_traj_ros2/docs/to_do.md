# To do 
- [x] Migrate SAS to UAV trajectory ROS2 
- [ ] Write a ROS2 node in cpp to provide waypoints based on location,heading, and leg constraints to get from one point to another
- [x] Visualize waypoints with marker on rviz2
- [ ] Refactor MPC code 
- [x] Have MPC trajectory visualized 
- [x] Figure out time synch between global planner and MPC planner
  - [x] Need to set the global planner to where I will be at in the next n seconds, to prevent lag input
- [ ] Improve MPC to have Velocity Control (right now its constant) 



# Help me
- Visualize markers: http://library.isr.ist.utl.pt/docs/roswiki/rviz(2f)Tutorials(2f)Markers(3a20)Basic(20)Shapes.html
- https://aviation.stackexchange.com/questions/2871/how-to-calculate-angular-velocity-and-radius-of-a-turn, turn radius     