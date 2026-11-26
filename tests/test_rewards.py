import math
import unittest
from dataclasses import replace

from config.environment_config import EnvironmentConfig
from config.reward_config import RewardConfig
from environment.collision import Outcome, classify_touchdown
from environment.physics_engine import PhysicsEngine, StepResult
from environment.rocket import RocketState, make_state
from training.reward_calculator import COMPONENT_NAMES, PROGRESS, RewardCalculator

ENV = EnvironmentConfig()
RC = RewardConfig()


def st(**kw):
    base = dict(x=0.0, y=50.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=400.0)
    base.update(kw)
    return RocketState(**base)


def flight(state, events=(), outcome=Outcome.IN_FLIGHT):
    return StepResult(state=state, outcome=outcome, done=outcome.is_terminal, events=tuple(events), touchdown=None)


def touchdown(**kw):
    base = dict(x=1.0, y=5.0, vx=0.2, vy=-1.0, theta=0.05, omega=0.1, fuel=300.0)
    base.update(kw)
    s = RocketState(**base)
    rep = classify_touchdown(s, PhysicsEngine(ENV).pad, ENV.criteria)
    # engine zeroes velocities after recording the report; mimic that
    rest = s.copy(); rest.vx = rest.vy = rest.omega = 0.0
    return StepResult(state=rest, outcome=rep.outcome, done=True, events=(), touchdown=rep)


def calc(cfg=RC, start=None):
    c = RewardCalculator(cfg, ENV)
    c.reset(start or st())
    return c


def stepto(c, state, **kw):
    return c.step(state, flight(state, **kw)).components


class TestProgressTerms(unittest.TestCase):
    def test_closer_is_positive_farther_negative(self):
        c = calc(start=st(x=30.0, y=80.0))
        self.assertGreater(stepto(c, st(x=29.0, y=79.0))["progress_distance"], 0)
        c = calc(start=st(x=30.0, y=80.0))
        self.assertLess(stepto(c, st(x=31.0, y=81.0))["progress_distance"], 0)

    def test_horizontal_error_reduction(self):
        c = calc(start=st(x=20.0))
        self.assertGreater(stepto(c, st(x=19.0))["progress_horizontal"], 0)

    def test_reducing_excess_descent_speed_is_positive(self):
        c = calc(start=st(y=10.0, vy=-14.0))
        self.assertGreater(stepto(c, st(y=10.0, vy=-8.0))["progress_descent_speed"], 0)

    def test_upright_and_spin_improvement(self):
        c = calc(start=st(theta=0.3, omega=0.8))
        comp = stepto(c, st(theta=0.2, omega=0.4))
        self.assertGreater(comp["progress_upright"], 0)
        self.assertGreater(comp["progress_angular"], 0)

    def test_no_progress_reward_for_standing_still(self):
        cfg = replace(RC, shaping_gamma=1.0)
        c = calc(cfg, start=st(x=10.0, theta=0.1))
        comp = stepto(c, st(x=10.0, theta=0.1))
        for k in PROGRESS:
            self.assertAlmostEqual(comp[k], 0.0, places=12)

    def test_progress_telescopes_round_trip(self):
        cfg = replace(RC, shaping_gamma=1.0)
        a, b = st(x=20.0, y=70.0, theta=0.2), st(x=5.0, y=30.0, theta=0.05, vy=-6.0)
        c = calc(cfg, start=a)
        total = sum(stepto(c, s)[k] for s in (b, a) for k in PROGRESS)
        self.assertAlmostEqual(total, 0.0, places=9)


class TestPenalties(unittest.TestCase):
    def test_no_penalty_inside_free_thresholds(self):
        comp = stepto(calc(), st(y=5.0, vx=1.0, theta=0.1, omega=0.2))
        self.assertEqual(comp["excess_horizontal_speed"], 0)
        self.assertEqual(comp["excess_tilt"], 0)
        self.assertEqual(comp["excess_rotation"], 0)

    def test_penalties_when_excessive(self):
        comp = stepto(calc(), st(y=5.0, vx=15.0, theta=0.55, omega=2.0))
        self.assertLess(comp["excess_horizontal_speed"], 0)
        self.assertLess(comp["excess_tilt"], 0)
        self.assertLess(comp["excess_rotation"], 0)

    def test_fuel_usage_proportional(self):
        c = calc(start=st(fuel=400.0))
        a = stepto(c, st(fuel=399.0))["fuel_usage"]
        c = calc(start=st(fuel=400.0))
        b = stepto(c, st(fuel=398.0))["fuel_usage"]
        self.assertLess(a, 0)
        self.assertAlmostEqual(b, 2 * a)

    def test_fuel_exhausted_event_penalised(self):
        comp = stepto(calc(), st(fuel=0.0), events=["FUEL_EXHAUSTED"])
        self.assertEqual(comp["fuel_exhausted"], -RC.fuel_exhausted_penalty)

    def test_time_penalty_each_step(self):
        self.assertEqual(stepto(calc(), st())["time_penalty"], -RC.time_penalty)


class TestTerminal(unittest.TestCase):
    def total(self, **kw):
        c = calc()
        res = touchdown(**kw)
        return c.step(res.state, res).total

    def test_success_large_positive(self):
        self.assertGreater(self.total(), 100)

    def test_every_failure_is_negative_and_below_success(self):
        success = self.total()
        cases = {"crash": dict(vy=-12.0), "side": dict(theta=0.8), "hard": dict(vy=-5.0),
                 "excessive": dict(vx=3.0), "off_pad": dict(x=20.0)}
        for name, kw in cases.items():
            t = self.total(**kw)
            self.assertLess(t, 0, name)
            self.assertLess(t, success, name)

    def test_crash_worse_than_off_pad_safe_touchdown(self):
        self.assertLess(self.total(vy=-12.0, x=20.0), self.total(x=20.0))

    def test_center_landing_beats_edge_landing(self):
        self.assertGreater(self.total(x=0.0), self.total(x=5.5))

    def test_softer_landing_scores_higher(self):
        self.assertGreater(self.total(vy=-0.5), self.total(vy=-2.9))

    def test_crash_step_gets_no_shaping_reward_from_zeroed_velocity(self):
        c = calc(start=st(y=20.0, vy=-25.0))
        res = touchdown(vy=-25.0)
        comp = c.step(res.state, res).components
        for k in PROGRESS:
            self.assertEqual(comp[k], 0.0)

    def test_out_of_bounds_and_timeout_negative(self):
        s = st(x=120.0)
        self.assertEqual(calc().step(s, flight(s, outcome=Outcome.OUT_OF_BOUNDS)).components["out_of_bounds"],
                         -RC.out_of_bounds_penalty)
        self.assertEqual(calc().step(st(), flight(st()), truncated=True).components["timeout"],
                         -RC.timeout_penalty)


class TestSeverityAndOrdering(unittest.TestCase):
    def total(self, **kw):
        c = calc()
        res = touchdown(**kw)
        return c.step(res.state, res).components

    def test_failed_touchdown_penalised_by_impact_speed(self):
        slow = self.total(vy=-4.0)["touchdown_severity"]
        fast = self.total(vy=-9.0)["touchdown_severity"]
        self.assertLess(fast, slow)
        self.assertLess(slow, 0)

    def test_severity_capped_and_absent_on_success(self):
        self.assertGreaterEqual(self.total(vy=-80.0)["touchdown_severity"], -RC.touchdown_severity_weight)
        self.assertEqual(self.total()["touchdown_severity"], 0.0)

    def test_hover_to_timeout_scores_worse_than_hard_landing(self):
        """Regression for the first training run: hovering (-33) used to beat a hard landing (-50)."""
        eng = PhysicsEngine(ENV, make_state(ENV.rocket, y=30.0))
        c = RewardCalculator(RC, ENV)
        c.reset(eng.state)
        total, g = 0.0, ENV.world.gravity
        eng.state.throttle = (ENV.rocket.dry_mass + ENV.rocket.fuel_capacity) * g / ENV.rocket.max_thrust
        for i in range(900):
            thr = (ENV.rocket.dry_mass + eng.state.fuel) * g / ENV.rocket.max_thrust
            res = eng.step(thr, 0.0)
            total += c.step(res.state, res, truncated=(i == 899)).total
        hard = calc().step(*(lambda r: (r.state, r))(touchdown(vy=-5.0))).total
        self.assertLess(total, hard)

    def test_success_beats_hard_beats_crash(self):
        succ = calc().step(*(lambda r: (r.state, r))(touchdown())).total
        hard = calc().step(*(lambda r: (r.state, r))(touchdown(vy=-5.0))).total
        crash = calc().step(*(lambda r: (r.state, r))(touchdown(vy=-14.0))).total
        self.assertGreater(succ, hard)
        self.assertGreater(hard, crash)


class TestStructure(unittest.TestCase):
    def test_all_components_always_present_and_sum_matches_total(self):
        bd = calc().step(st(x=1.0), flight(st(x=1.0)))
        self.assertEqual(set(bd.components), set(COMPONENT_NAMES))
        self.assertAlmostEqual(bd.total, sum(bd.components.values()))

    def test_weights_are_configurable(self):
        cfg = replace(RC, time_penalty=0.0, distance_weight=0.0)
        c = calc(cfg, start=st(x=30.0))
        comp = stepto(c, st(x=29.0))
        self.assertEqual(comp["progress_distance"], 0.0)
        self.assertEqual(comp["time_penalty"], 0.0)

    def test_step_before_reset_raises(self):
        with self.assertRaises(RuntimeError):
            RewardCalculator(RC, ENV).step(st(), flight(st()))


class TestLoopholes(unittest.TestCase):
    def test_hovering_is_unprofitable(self):
        eng = PhysicsEngine(ENV, make_state(ENV.rocket, y=60.0))
        calc_ = RewardCalculator(RC, ENV)
        calc_.reset(eng.state)
        hover = (ENV.rocket.dry_mass + ENV.rocket.fuel_capacity) * ENV.world.gravity / ENV.rocket.max_thrust
        eng.state.throttle = hover
        total = 0.0
        for _ in range(300):
            res = eng.step(hover, 0.0)
            total += calc_.step(res.state, res).total
            hover = (ENV.rocket.dry_mass + eng.state.fuel) * ENV.world.gravity / ENV.rocket.max_thrust
        self.assertLess(total, 0.0)

    def test_shaping_gamma_below_one_would_create_hover_loophole(self):
        """Regression: documents why shaping_gamma must stay 1.0."""
        eng = PhysicsEngine(ENV, make_state(ENV.rocket, y=60.0))
        calc_ = RewardCalculator(replace(RC, shaping_gamma=0.99), ENV)
        calc_.reset(eng.state)
        g = ENV.world.gravity
        eng.state.throttle = (ENV.rocket.dry_mass + ENV.rocket.fuel_capacity) * g / ENV.rocket.max_thrust
        total = 0.0
        for _ in range(100):
            thr = (ENV.rocket.dry_mass + eng.state.fuel) * g / ENV.rocket.max_thrust
            res = eng.step(thr, 0.0)
            total += calc_.step(res.state, res).components["progress_distance"]
        self.assertGreater(total, 0.0)   # standing still was paid -> default must not use this

    def test_default_shaping_gamma_is_one(self):
        self.assertEqual(RewardConfig().shaping_gamma, 1.0)

    def test_crashing_is_worse_than_hovering_until_timeout(self):
        crash = calc().step(*(lambda r: (r.state, r))(touchdown(vy=-20.0))).total
        self.assertLess(crash, -RC.timeout_penalty)


if __name__ == "__main__":
    unittest.main()
