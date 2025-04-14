
(cl:in-package :asdf)

(defsystem "franka_h2-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "TrajectoryData" :depends-on ("_package_TrajectoryData"))
    (:file "_package_TrajectoryData" :depends-on ("_package"))
  ))