cmake_minimum_required(VERSION 3.0.2)
project(grasp_scheduler)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
)

catkin_python_setup()

# add_message_files(
#   FILES
# )

# add_service_files(
#   FILES
# )

# generate_messages(
#   DEPENDENCIES
# )

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
