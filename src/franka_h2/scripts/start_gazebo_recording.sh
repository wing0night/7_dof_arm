#!/bin/bash
# 启动 Gazebo 并开始录制
gz service -s /gazebo/record_video call '{
  "start": true,
  "format": "mp4",
  "save_path": "/home/night/videos/output.mp4"
}'

# 等待录制时长
sleep 15

# 停止录制
gz service -s /gazebo/record_video call '{"start": false}'