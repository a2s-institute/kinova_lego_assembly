# One segmented instance returned by the LegoSegmentation service.
int32 obj_id
int32 class_id
string class_name
float32 score
# Absolute pixel bounding box [x1, y1, x2, y2].
float32[4] bbox_xyxy
# Normalized bounding box in [0, 1].
float32[4] bbox_xyxyn
# Mask outline in absolute pixels; empty when the model has no masks.
geometry_msgs/Point32[] mask_polygon_xy
