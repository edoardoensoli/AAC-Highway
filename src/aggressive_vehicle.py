from highway_env.vehicle.behavior import IDMVehicle


class AggressiveIDMVehicle(IDMVehicle):
    ACC_MAX = 5.0
    ACC_MIN = -3.5
    tau = 0.8
    delta = 3.0
    POLITENESS = 0.8
    LANE_CHANGE_MIN_ACC_GAIN = 0.1
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0