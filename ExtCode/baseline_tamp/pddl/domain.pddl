(define (domain stp_traditional_tamp)
  (:requirements :strips :typing :negative-preconditions :equality :quantified-preconditions)
  (:types
    pose package grasp traj size
  )

  (:predicates
    (AtRobot ?p - pose)
    (AtPkg ?pkg - package ?p - pose)
    (Holding ?pkg - package)
    (HandEmpty)
    (Picked ?pkg - package)
    (Urgent ?pkg - package)
    (NonUrgent ?pkg - package)
    (UrgentPhaseOpen)
    (SizeSmall ?pkg - package)
    (SizeMedium ?pkg - package)
    (SizeLarge ?pkg - package)
    (CarriedSize ?s - size)
    (Motion ?p1 - pose ?p2 - pose ?t - traj)
    (MotionLoaded ?p1 - pose ?p2 - pose ?t - traj ?s - size)
    (CanPick ?pkg - package ?p - pose ?g - grasp)
    (CanPlace ?pkg - package ?p - pose)
  )

  (:action move
    :parameters (?from - pose ?to - pose ?traj - traj)
    :precondition (and
      (AtRobot ?from)
      (Motion ?from ?to ?traj)
    )
    :effect (and
      (AtRobot ?to)
      (not (AtRobot ?from))
    )
  )

  (:action move_loaded
    :parameters (?from - pose ?to - pose ?traj - traj ?s - size ?pkg - package)
    :precondition (and
      (AtRobot ?from)
      (Holding ?pkg)
      (CarriedSize ?s)
      (MotionLoaded ?from ?to ?traj ?s)
    )
    :effect (and
      (AtRobot ?to)
      (not (AtRobot ?from))
    )
  )

  (:action pick_urgent
    :parameters (?pkg - package ?p - pose ?g - grasp)
    :precondition (and
      (UrgentPhaseOpen)
      (Urgent ?pkg)
      (AtRobot ?p)
      (AtPkg ?pkg ?p)
      (CanPick ?pkg ?p ?g)
      (HandEmpty)
    )
    :effect (and
      (Holding ?pkg)
      (not (AtPkg ?pkg ?p))
      (not (HandEmpty))
    )
  )

  (:action close_urgent_phase
    :parameters ()
    :precondition (and
      (UrgentPhaseOpen)
      (forall (?u - package)
        (or (not (Urgent ?u)) (Picked ?u))
      )
    )
    :effect (and
      (not (UrgentPhaseOpen))
    )
  )

  (:action pick_normal
    :parameters (?pkg - package ?p - pose ?g - grasp)
    :precondition (and
      (NonUrgent ?pkg)
      (not (UrgentPhaseOpen))
      (AtRobot ?p)
      (AtPkg ?pkg ?p)
      (CanPick ?pkg ?p ?g)
      (HandEmpty)
    )
    :effect (and
      (Holding ?pkg)
      (not (AtPkg ?pkg ?p))
      (not (HandEmpty))
    )
  )

  (:action place
    :parameters (?pkg - package ?p - pose)
    :precondition (and
      (Holding ?pkg)
      (AtRobot ?p)
      (CanPlace ?pkg ?p)
    )
    :effect (and
      (Picked ?pkg)
      (HandEmpty)
      (not (Holding ?pkg))
    )
  )
)
