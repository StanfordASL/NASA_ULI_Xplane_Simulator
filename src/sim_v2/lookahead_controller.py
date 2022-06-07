import time

import numpy as np

from utils import sim_data

wrap_pi = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi
angle_diff = lambda x, y: wrap_pi(x - y)


class LookaheadController:

    def __init__(self, client, v_max=10, lookahead=20, dt=0.1, *args, **kwargs):
        self.client = client
        self.controller_state = "tracking"
        self.v_max = v_max
        self.lookahead = lookahead
        self.dt = dt

        self.parkbrake = 0
        self.steering = 0
        self.throttle = 0
        self.speed_error_integrator = 0

    def get_control(self, belief, trajectory):
        del belief  # unused until belief contains heading/velocity information.
        client, v_max, lookahead, dt = self.client, self.v_max, self.lookahead, self.dt

        state = sim_data.TaxiState.get_from_sim(client)
        s, d = trajectory.frenet_coordinates((state.e, state.n))

        # Lookahead.
        enqk_upcoming = trajectory(np.minimum(s + np.arange(lookahead), trajectory.en.t[-1]))
        k_max = enqk_upcoming[np.argmax(np.abs(enqk_upcoming[:, 3])), 3]
        v_target = np.minimum(0.8 * v_max, np.sqrt(1 / np.maximum(np.abs(k_max), 1e-6)))
        heading_error = angle_diff(state.q, np.mean(enqk_upcoming[:, 2]))
        _, lookahead_d = trajectory.frenet_coordinates(
            np.array([state.e, state.n]) + lookahead * np.array([np.cos(state.q), np.sin(state.q)]))

        if self.controller_state == "turn_in_place":
            # Controller state transition(s).
            if np.abs(heading_error) < np.pi / 8:
                self.controller_state = "tracking"
                self.speed_error_integrator = 0
                return self.get_control(None, trajectory)

            self.steering = -0.5 * np.sign(heading_error)
            if state.v_x > 0.5:
                # Stop.
                self.parkbrake, self.throttle = 1, 0.5
            else:
                # Turn in place.
                self.parkbrake = 0
                if np.abs(state.r) > 0.1:
                    self.throttle = 0.5
                else:
                    self.throttle += 0.01

        elif self.controller_state == "tracking":
            # Controller state transition(s).
            if np.abs(heading_error) > np.pi / 4:
                self.controller_state = "turn_in_place"
                self.throttle = 0.5
                return self.get_control(None, trajectory)
            if s > trajectory.en.t[4] - 5:
                self.controller_state = "done"
                return self.get_control(None, trajectory)

            # Longitudinal control.
            speed_error = state.v_x - v_target
            self.speed_error_integrator = np.clip(self.speed_error_integrator + dt * speed_error, -2, 2)
            if state.v_x > v_target / 0.8:
                self.parkbrake, self.throttle = 0.1, 0
            else:
                self.parkbrake = 0
                self.throttle = -1e-1 * speed_error - 1e-1 * self.speed_error_integrator

            # Lateral control.
            self.steering = np.clip(k_max - 1e-1 * (d + lookahead_d), -0.3, 0.3)

        elif self.controller_state == "done":
            self.parkbrake, self.steering, self.throttle = 1, 0, 0

        sim_data.TaxiControl(self.parkbrake, self.steering, self.throttle).send_to_sim(client)
        time.sleep(dt)
        return None, (self.controller_state == "done")
