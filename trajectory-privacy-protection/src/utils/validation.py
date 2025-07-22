from pydantic import BaseModel, conlist, validator
from typing import List, Tuple

class TrajectoryPoint(BaseModel):
    longitude: float
    latitude: float

class TrajectoryValidation:
    @staticmethod
    def validate_trajectory(trajectory: List[TrajectoryPoint]) -> bool:
        if not trajectory:
            raise ValueError("Trajectory cannot be empty.")
        
        for point in trajectory:
            if not (-180 <= point.longitude <= 180):
                raise ValueError(f"Invalid longitude: {point.longitude}. Must be between -180 and 180.")
            if not (-90 <= point.latitude <= 90):
                raise ValueError(f"Invalid latitude: {point.latitude}. Must be between -90 and 90.")
        
        return True

    @staticmethod
    def validate_start_end_points(trajectory: List[TrajectoryPoint]) -> Tuple[TrajectoryPoint, TrajectoryPoint]:
        if len(trajectory) < 2:
            raise ValueError("Trajectory must have at least two points for start and end validation.")
        
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        return start_point, end_point

    @staticmethod
    def validate_continuity(trajectory: List[TrajectoryPoint]) -> bool:
        for i in range(1, len(trajectory)):
            if trajectory[i].longitude == trajectory[i-1].longitude and trajectory[i].latitude == trajectory[i-1].latitude:
                raise ValueError("Trajectory points must be distinct.")
        
        return True