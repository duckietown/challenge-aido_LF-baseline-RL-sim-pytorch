import numpy as np

class PurePursuitExpert:
    def __init__(self, env, ref_velocity=0.8, position_threshold=0.04,
                 following_distance=0.4, gain=10, max_iterations=1000):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold
        self.gain=gain

    def predict(self, observation):  # we don't really care about the observation for this implementation
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        """
        my_dir = self.env.get_dir_vec()
        def angle(v1=my_dir, v2=closest_tangent):
            x1 = v1[0]
            y1 = v1[2]
            x2 = v2[0]
            y2 = v2[2]

            return np.arctan2(y2, x2) - np.arctan2(y1, x1)
        if np.abs(angle()) < self.position_threshold:
            return self.ref_velocity, 0
        """

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = self.gain * -dot

        return self.ref_velocity, steering

class StanleyExpert:
    def __init__(self, env, ref_velocity=0.8, position_threshold=0.04,
                 following_distance=10, gain=10, max_iterations=1000):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold
        self.gain=gain

    def predict(self, observation):  # we don't really care about the observation for this implementation
        # Here, the closest_tangent is actually the trajectory of the curve that's closest to the car
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        my_dir = self.env.get_dir_vec()

        def angle(v1,v2):
            x1 = v1[0]
            y1 = v1[2]
            x2 = v2[0]
            y2 = v2[2]

            return np.arctan2(y2, x2) - np.arctan2(y1, x1)

            cosang = np.dot(v1, v2)
            sinang = np.linalg.norm(np.cross(v1, v2))
            return np.arctan2(sinang, cosang)

            u1 = v1 / np.linalg.norm(v1)
            u2 = v2 / np.linalg.norm(v2)

            dot_product = np.dot(u1, u2)
            return np.arccos(np.clip(dot_product, -1, 1))

        new_angle = angle(my_dir, closest_tangent)
        #return self.ref_velocity, new_angle

        #print(angle(my_dir, closest_tangent))

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        smol = 0.000001
        new_angle += angle(closest_point, (curve_point+[smol, smol, smol]))
        print(new_angle)
        return self.ref_velocity, new_angle
