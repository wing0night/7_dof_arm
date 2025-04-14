;; Auto-generated. Do not edit!


(when (boundp 'franka_h2::TrajectoryData)
  (if (not (find-package "FRANKA_H2"))
    (make-package "FRANKA_H2"))
  (shadow 'TrajectoryData (find-package "FRANKA_H2")))
(unless (find-package "FRANKA_H2::TRAJECTORYDATA")
  (make-package "FRANKA_H2::TRAJECTORYDATA"))

(in-package "ROS")
;;//! \htmlinclude TrajectoryData.msg.html


(defclass franka_h2::TrajectoryData
  :super ros::object
  :slots (_joint_names _positions _velocities _stamp ))

(defmethod franka_h2::TrajectoryData
  (:init
   (&key
    ((:joint_names __joint_names) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:positions __positions) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:velocities __velocities) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:stamp __stamp) (instance ros::time :init))
    )
   (send-super :init)
   (setq _joint_names __joint_names)
   (setq _positions __positions)
   (setq _velocities __velocities)
   (setq _stamp __stamp)
   self)
  (:joint_names
   (&optional __joint_names)
   (if __joint_names (setq _joint_names __joint_names)) _joint_names)
  (:positions
   (&optional __positions)
   (if __positions (setq _positions __positions)) _positions)
  (:velocities
   (&optional __velocities)
   (if __velocities (setq _velocities __velocities)) _velocities)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:serialization-length
   ()
   (+
    ;; string[] _joint_names
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _joint_names)) 4
    ;; float64[] _positions
    (* 8    (length _positions)) 4
    ;; float64[] _velocities
    (* 8    (length _velocities)) 4
    ;; time _stamp
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _joint_names
     (write-long (length _joint_names) s)
     (dolist (elem _joint_names)
       (write-long (length elem) s) (princ elem s)
       )
     ;; float64[] _positions
     (write-long (length _positions) s)
     (dotimes (i (length _positions))
       (sys::poke (elt _positions i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _velocities
     (write-long (length _velocities) s)
     (dotimes (i (length _velocities))
       (sys::poke (elt _velocities i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string[] _joint_names
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _joint_names (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _joint_names i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; float64[] _positions
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _positions (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _positions i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _velocities
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _velocities (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _velocities i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get franka_h2::TrajectoryData :md5sum-) "4fd91ba6a794e307aefbfd0f95764a4e")
(setf (get franka_h2::TrajectoryData :datatype-) "franka_h2/TrajectoryData")
(setf (get franka_h2::TrajectoryData :definition-)
      "string[] joint_names
float64[] positions
float64[] velocities
time stamp
")



(provide :franka_h2/TrajectoryData "4fd91ba6a794e307aefbfd0f95764a4e")


