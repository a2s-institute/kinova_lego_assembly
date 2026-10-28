cmake_minimum_required(VERSION 3.8)
project(kinova_lego_assembly_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/DetectedObject.msg
  msg/AssemblyStatus.msg
  srv/SegmentImage.srv
  srv/ClearPlacementRecord.srv
  action/AssembleModel.action
  DEPENDENCIES geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
