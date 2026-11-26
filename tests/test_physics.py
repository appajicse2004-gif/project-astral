"""Physics tests. Plain unittest, so they also run under pytest."""
import math
import unittest

from config.environment_config import EnvironmentConfig, G0
from environment.collision import Outcome, classify_touchdown
from environment.landing_pad import LandingPad
from environment.physics_engine import PhysicsEngine
from environment.rocket import (RocketState, hull_corners, lowest_point_height,
                                make_state, moment_of_inertia, thrust_direction,
                                total_mass)

CFG = EnvironmentConfig()
R, W = CFG.rocket, CFG.world


def engine(**kw):
    return PhysicsEngine(CFG, make_state(R, **kw))


def run(eng, n, throttle=0.0, rot=0.0):
    res = None
    for _ in range(n):
        res = eng.step(throttle, rot)
        if res.done:
            break
    return res


class TestMassAndGeometry(unittest.TestCase):
    def test_total_mass_includes_fuel(self):
        s = make_state(R, fuel_fraction=0.5)
        self.assertAlmostEqual(total_mass(s, R), R.dry_mass + 0.5 * R.fuel_capacity)

    def test_moment_of_inertia(self):
        self.assertAlmostEqual(moment_of_inertia(1200.0, R), 1200.0 * R.length ** 2 / 12)

    def test_thrust_direction(self):
        self.assertEqual(thrust_direction(0.0), (-0.0, 1.0))
        dx, dy = thrust_direction(math.pi / 2)
        self.assertAlmostEqual(dx, -1.0)
        self.assertAlmostEqual(dy, 0.0)

    def test_lowest_point(self):
        s = RocketState(y=10.0, theta=0.0)
        self.assertAlmostEqual(lowest_point_height(s, R), 10.0 - R.length / 2)
        s.theta = math.pi / 2          # lying on its side
        self.assertAlmostEqual(lowest_point_height(s, R), 10.0 - R.width / 2)

    def test_hull_corners_upright(self):
        pts = hull_corners(RocketState(x=1.0, y=10.0), R)
        ys = sorted(p[1] for p in pts)
        self.assertAlmostEqual(ys[0], 10.0 - R.length / 2)
        self.assertAlmostEqual(ys[-1], 10.0 + R.length / 2)


class TestForces(unittest.TestCase):
    def test_free_fall_matches_analytic(self):
        e = engine(y=100.0)
        n = 60
        run(e, n)
        t = n * W.dt
        self.assertAlmostEqual(e.state.vy, -W.gravity * t, places=9)
        self.assertAlmostEqual(e.state.y, 100.0 - 0.5 * W.gravity * t * t, places=9)
        self.assertAlmostEqual(e.state.vx, 0.0)

    def test_hover_throttle_cancels_gravity(self):
        m = total_mass(make_state(R), R)
        hover = m * W.gravity / R.max_thrust
        e = engine(y=100.0)
        e.state.throttle = hover            # start already at hover throttle
        e.step(hover, 0.0)
        self.assertAlmostEqual(e.state.ay, 0.0, places=9)
        self.assertAlmostEqual(e.state.vy, 0.0, places=9)

    def test_tilted_thrust_accelerates_sideways(self):
        theta = 0.3
        e = engine(y=100.0, theta=theta)
        e.state.throttle = 1.0
        e.step(1.0, 0.0)
        m = total_mass(make_state(R), R)
        self.assertAlmostEqual(e.state.ax, -R.max_thrust * math.sin(theta) / m, places=9)
        self.assertLess(e.state.ax, 0.0)    # +theta (CCW) pushes toward -x
        self.assertAlmostEqual(e.state.ay,
                               R.max_thrust * math.cos(theta) / m - W.gravity, places=9)

    def test_throttle_is_rate_limited(self):
        e = engine(y=100.0)
        e.step(1.0, 0.0)
        self.assertAlmostEqual(e.state.throttle, R.throttle_rate * W.dt)

    def test_commands_are_clipped(self):
        e = engine(y=100.0)
        e.state.throttle = 1.0
        e.step(5.0, 9.0)
        self.assertLessEqual(e.state.throttle, 1.0)
        m = total_mass(make_state(R), R)
        self.assertAlmostEqual(e.state.alpha, R.max_torque / moment_of_inertia(m, R), places=9)


class TestRotation(unittest.TestCase):
    def test_torque_gives_angular_acceleration(self):
        e = engine(y=100.0)
        e.step(0.0, 1.0)
        m = total_mass(make_state(R), R)
        alpha = R.max_torque / moment_of_inertia(m, R)
        self.assertAlmostEqual(e.state.alpha, alpha, places=9)
        self.assertAlmostEqual(e.state.omega, alpha * W.dt, places=9)
        self.assertGreater(e.state.omega, 0.0)

    def test_negative_command_rotates_clockwise(self):
        e = engine(y=100.0)
        run(e, 10, rot=-1.0)
        self.assertLess(e.state.omega, 0.0)
        self.assertLess(e.state.theta, 0.0)

    def test_free_rotation_conserves_omega(self):
        e = engine(y=100.0, omega=0.2)
        run(e, 30)
        self.assertAlmostEqual(e.state.omega, 0.2, places=12)

    def test_angle_wraps(self):
        e = engine(y=100.0, theta=3.1, omega=1.0)
        run(e, 30)
        self.assertLessEqual(abs(e.state.theta), math.pi)


class TestFuel(unittest.TestCase):
    def test_burn_rate_matches_isp(self):
        e = engine(y=100.0)
        e.state.throttle = 1.0
        f0 = e.state.fuel
        run(e, 60, throttle=1.0)             # exactly 1.0 s of full throttle
        expected = R.max_thrust / (R.isp * G0) * 1.0
        self.assertAlmostEqual(f0 - e.state.fuel, expected, places=9)

    def test_no_fuel_use_when_coasting(self):
        e = engine(y=100.0)
        run(e, 30)
        self.assertAlmostEqual(e.state.fuel, R.fuel_capacity)

    def test_rcs_uses_fuel(self):
        e = engine(y=100.0)
        run(e, 60, rot=1.0)
        self.assertAlmostEqual(R.fuel_capacity - e.state.fuel, R.rcs_max_flow * 1.0, places=9)

    def test_mass_decreases_as_fuel_burns(self):
        e = engine(y=100.0)
        e.state.throttle = 1.0
        m0 = total_mass(e.state, R)
        run(e, 30, throttle=1.0)
        self.assertLess(total_mass(e.state, R), m0)

    def test_fuel_exhaustion_cuts_thrust(self):
        e = engine(y=100.0, fuel_fraction=0.01 / R.fuel_capacity)  # 0.01 kg
        e.state.throttle = 1.0
        r1 = e.step(1.0, 0.0)
        self.assertIn("FUEL_EXHAUSTED", r1.events)
        self.assertEqual(e.state.fuel, 0.0)
        e.step(1.0, 1.0)
        self.assertAlmostEqual(e.state.ay, -W.gravity, places=9)   # engine dead
        self.assertAlmostEqual(e.state.alpha, 0.0, places=12)      # RCS dead
        self.assertEqual(e.events.count("FUEL_EXHAUSTED"), 1)

    def test_fuel_never_negative(self):
        e = engine(y=100.0, fuel_fraction=0.001)
        e.state.throttle = 1.0
        run(e, 100, throttle=1.0, rot=1.0)
        self.assertGreaterEqual(e.state.fuel, 0.0)


class TestTouchdownClassification(unittest.TestCase):
    pad = LandingPad(CFG.pad)
    crit = CFG.criteria

    def cls(self, **kw):
        base = dict(x=1.0, y=5.0, vx=0.2, vy=-1.0, theta=0.05, omega=0.1)
        base.update(kw)
        return classify_touchdown(RocketState(**base), self.pad, self.crit)

    def test_success(self):
        rep = self.cls()
        self.assertEqual(rep.outcome, Outcome.SUCCESS)
        self.assertEqual(rep.failed_criteria, ())

    def test_touching_ground_alone_is_not_success(self):
        self.assertNotEqual(self.cls(vy=-5.0).outcome, Outcome.SUCCESS)
        self.assertNotEqual(self.cls(theta=0.3).outcome, Outcome.SUCCESS)
        self.assertNotEqual(self.cls(x=30.0).outcome, Outcome.SUCCESS)
        self.assertNotEqual(self.cls(vx=4.0).outcome, Outcome.SUCCESS)
        self.assertNotEqual(self.cls(omega=1.0).outcome, Outcome.SUCCESS)

    def test_hard_landing(self):
        rep = self.cls(vy=-5.0)
        self.assertEqual(rep.outcome, Outcome.HARD_LANDING)
        self.assertIn("vertical_speed", rep.failed_criteria)

    def test_crash(self):
        self.assertEqual(self.cls(vy=-12.0).outcome, Outcome.CRASH)

    def test_side_collision(self):
        self.assertEqual(self.cls(theta=0.8).outcome, Outcome.SIDE_COLLISION)

    def test_excessive_horizontal_velocity(self):
        rep = self.cls(vx=3.0)
        self.assertEqual(rep.outcome, Outcome.EXCESSIVE_VELOCITY)
        self.assertIn("horizontal_speed", rep.failed_criteria)

    def test_excessive_angular_velocity(self):
        self.assertEqual(self.cls(omega=0.8).outcome, Outcome.EXCESSIVE_VELOCITY)

    def test_off_pad_when_otherwise_safe(self):
        rep = self.cls(x=20.0)
        self.assertEqual(rep.outcome, Outcome.OFF_PAD)
        self.assertFalse(rep.on_pad)
        self.assertAlmostEqual(rep.pad_offset, 20.0)

    def test_pad_edge_is_inclusive(self):
        self.assertTrue(self.pad.contains(CFG.pad.half_width))
        self.assertFalse(self.pad.contains(CFG.pad.half_width + 0.01))


class TestEpisodeFlow(unittest.TestCase):
    def test_gentle_drop_onto_pad_succeeds(self):
        # Rocket released just above the ground with a small downward speed.
        e = engine(y=R.length / 2 + 0.05, vy=-1.0)
        res = run(e, 600)
        self.assertTrue(res.done)
        self.assertEqual(res.outcome, Outcome.SUCCESS)
        self.assertAlmostEqual(lowest_point_height(e.state, R), 0.0, places=9)

    def test_fast_fall_crashes(self):
        e = engine(y=30.0, vy=-20.0)
        res = run(e, 600)
        self.assertEqual(res.outcome, Outcome.CRASH)
        self.assertIsNotNone(res.touchdown)
        self.assertGreater(res.touchdown.vertical_speed, CFG.criteria.crash_vertical_speed)

    def test_free_fall_from_height_crashes(self):
        res = run(engine(y=100.0), 2000)
        self.assertEqual(res.outcome, Outcome.CRASH)

    def test_landing_off_pad(self):
        e = engine(x=40.0, y=R.length / 2 + 0.05, vy=-1.0)
        self.assertEqual(run(e, 600).outcome, Outcome.OFF_PAD)

    def test_out_of_bounds_sideways(self):
        e = engine(x=99.0, y=100.0, vx=20.0)
        self.assertEqual(run(e, 600).outcome, Outcome.OUT_OF_BOUNDS)

    def test_out_of_bounds_ceiling(self):
        e = engine(y=140.0, vy=30.0)
        self.assertEqual(run(e, 600).outcome, Outcome.OUT_OF_BOUNDS)

    def test_step_after_done_raises(self):
        e = engine(y=R.length / 2 + 0.01, vy=-1.0)
        run(e, 600)
        with self.assertRaises(RuntimeError):
            e.step(0.0, 0.0)

    def test_reset_restores_flight(self):
        e = engine(y=R.length / 2 + 0.01, vy=-1.0)
        run(e, 600)
        e.reset(make_state(R, y=50.0))
        self.assertFalse(e.done)
        self.assertEqual(e.state.y, 50.0)
        self.assertEqual(e.events, [])

    def test_deterministic(self):
        def trajectory():
            e = engine(x=10.0, y=100.0, vx=-1.0, theta=0.1)
            out = []
            for i in range(120):
                e.step(0.6 + 0.1 * math.sin(i / 10), math.cos(i / 7))
                out.append((e.state.x, e.state.y, e.state.theta, e.state.fuel))
            return out
        self.assertEqual(trajectory(), trajectory())

    def test_config_serialises(self):
        self.assertIn("gravity", CFG.to_json())


if __name__ == "__main__":
    unittest.main()
