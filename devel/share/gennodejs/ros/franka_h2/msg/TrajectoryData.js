// Auto-generated. Do not edit!

// (in-package franka_h2.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class TrajectoryData {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.joint_names = null;
      this.positions = null;
      this.velocities = null;
      this.torques = null;
      this.stamp = null;
    }
    else {
      if (initObj.hasOwnProperty('joint_names')) {
        this.joint_names = initObj.joint_names
      }
      else {
        this.joint_names = [];
      }
      if (initObj.hasOwnProperty('positions')) {
        this.positions = initObj.positions
      }
      else {
        this.positions = [];
      }
      if (initObj.hasOwnProperty('velocities')) {
        this.velocities = initObj.velocities
      }
      else {
        this.velocities = [];
      }
      if (initObj.hasOwnProperty('torques')) {
        this.torques = initObj.torques
      }
      else {
        this.torques = [];
      }
      if (initObj.hasOwnProperty('stamp')) {
        this.stamp = initObj.stamp
      }
      else {
        this.stamp = {secs: 0, nsecs: 0};
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TrajectoryData
    // Serialize message field [joint_names]
    bufferOffset = _arraySerializer.string(obj.joint_names, buffer, bufferOffset, null);
    // Serialize message field [positions]
    bufferOffset = _arraySerializer.float64(obj.positions, buffer, bufferOffset, null);
    // Serialize message field [velocities]
    bufferOffset = _arraySerializer.float64(obj.velocities, buffer, bufferOffset, null);
    // Serialize message field [torques]
    bufferOffset = _arraySerializer.float64(obj.torques, buffer, bufferOffset, null);
    // Serialize message field [stamp]
    bufferOffset = _serializer.time(obj.stamp, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TrajectoryData
    let len;
    let data = new TrajectoryData(null);
    // Deserialize message field [joint_names]
    data.joint_names = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [positions]
    data.positions = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [velocities]
    data.velocities = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [torques]
    data.torques = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [stamp]
    data.stamp = _deserializer.time(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.joint_names.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    length += 8 * object.positions.length;
    length += 8 * object.velocities.length;
    length += 8 * object.torques.length;
    return length + 24;
  }

  static datatype() {
    // Returns string type for a message object
    return 'franka_h2/TrajectoryData';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'f137a938aa3a1c7bf0c5b969f6d8d555';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string[] joint_names
    float64[] positions
    float64[] velocities
    float64[] torques
    time stamp
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TrajectoryData(null);
    if (msg.joint_names !== undefined) {
      resolved.joint_names = msg.joint_names;
    }
    else {
      resolved.joint_names = []
    }

    if (msg.positions !== undefined) {
      resolved.positions = msg.positions;
    }
    else {
      resolved.positions = []
    }

    if (msg.velocities !== undefined) {
      resolved.velocities = msg.velocities;
    }
    else {
      resolved.velocities = []
    }

    if (msg.torques !== undefined) {
      resolved.torques = msg.torques;
    }
    else {
      resolved.torques = []
    }

    if (msg.stamp !== undefined) {
      resolved.stamp = msg.stamp;
    }
    else {
      resolved.stamp = {secs: 0, nsecs: 0}
    }

    return resolved;
    }
};

module.exports = TrajectoryData;
