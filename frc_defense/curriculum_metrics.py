"""Physical-piece lifecycle accounting for draft offense scenarios.

This evaluator uses simulator truth for outcomes only. It supplies no teacher
observations. Rewards are consumable increments, never qualification evidence.
"""
from __future__ import annotations
import math
import torch


class CurriculumMetrics:
    def __init__(self, env):
        self.env = env
        shape = env.piece_owner.shape
        def flags():
            return torch.zeros(shape, device=env.device, dtype=torch.bool)
        self.acquisition_credit = flags()
        self.external = flags()
        self.ferry_credit = flags()
        self.pending = flags()
        self.initial_staged = flags()
        self.inactive_origin = flags()
        self.required = env.piece_active.clone()
        self.required &= (env.piece_owner < 0) | (env.piece_owner == 3)
        self.external |= (env.piece_owner == 3) & (env.piece_pos[..., 0] < self.home_x)
        self.settling = torch.zeros(shape, device=env.device)
        self.lifecycle = torch.zeros_like(env.piece_owner)
        self.piece_batch = torch.zeros_like(env.piece_owner)
        self.batch_id = torch.zeros(env.n, device=env.device, dtype=torch.long)
        self.delivery_open = torch.zeros(env.n, device=env.device, dtype=torch.bool)
        # A new batch needs an intervening empty hopper. There cannot be more
        # completed batches than physics ticks within this bounded episode.
        self.batch_counts = torch.zeros((env.n, env.horizon + 1), device=env.device)
        self.score_batch_counts = torch.zeros_like(self.batch_counts)
        self.ferry_batch_counts = torch.zeros_like(self.batch_counts)
        self.batch_ended = torch.zeros_like(self.batch_counts, dtype=torch.bool)
        self.batch_completed = torch.zeros_like(self.batch_ended)
        self.acquisitions = torch.zeros(env.n, device=env.device)
        self.scores = torch.zeros_like(self.acquisitions)
        self.deliveries = torch.zeros_like(self.acquisitions)
        self.team_conversions = torch.zeros_like(self.acquisitions)
        self.staged_conversions = torch.zeros_like(self.acquisitions)
        self.initial_staged_conversions = torch.zeros_like(self.acquisitions)
        self.inactive_conversions = torch.zeros_like(self.acquisitions)
        self.inactive_deliveries = torch.zeros_like(self.acquisitions)
        self.post_boundary_scores = torch.zeros_like(self.acquisitions)
        self.post_boundary_deliveries = torch.zeros_like(self.acquisitions)
        self.encounter_ticks = torch.zeros_like(self.acquisitions)
        self.nearest_defender = torch.full_like(self.acquisitions, math.hypot(env.field_length, env.sim.field_width))
        self.elapsed = torch.zeros_like(self.acquisitions)
        self.active_seconds = torch.zeros_like(self.acquisitions)
        self.inactive_seconds = torch.zeros_like(self.acquisitions)
        self.delta = torch.zeros_like(self.acquisitions)
        self.route_seen = torch.zeros(env.n, device=env.device, dtype=torch.bool)
        self.previous_x = env.sim.pose[:, 3, 0].clone()
        self.current_staged = torch.zeros_like(self.acquisitions)
        self.inactive_staged = torch.zeros_like(self.acquisitions)

    def reset(self, mask):
        """Initialize episode metrics for selected worlds in the existing object."""
        env = self.env
        if (not isinstance(mask, torch.Tensor) or mask.device != env.device or
                mask.dtype is not torch.bool or tuple(mask.shape) != (env.n,)):
            raise ValueError("curriculum metric reset mask must be device bool [world]")

        def write_rows(tensor, values):
            row_mask = mask.reshape((env.n,) + (1,) * (tensor.ndim - 1))
            tensor.copy_(torch.where(row_mask, values, tensor))

        zero_fields = (
            "acquisition_credit", "ferry_credit", "pending", "initial_staged",
            "inactive_origin", "settling", "lifecycle", "piece_batch", "batch_id",
            "delivery_open", "batch_counts", "score_batch_counts", "ferry_batch_counts",
            "batch_ended", "batch_completed", "acquisitions", "scores", "deliveries",
            "team_conversions", "staged_conversions", "initial_staged_conversions",
            "inactive_conversions", "inactive_deliveries", "post_boundary_scores",
            "post_boundary_deliveries", "encounter_ticks", "elapsed", "active_seconds",
            "inactive_seconds", "delta", "route_seen", "current_staged", "inactive_staged",
        )
        for name in zero_fields:
            tensor = getattr(self, name)
            write_rows(tensor, torch.zeros_like(tensor))
        required = env.piece_active & ((env.piece_owner < 0) | (env.piece_owner == 3))
        write_rows(self.required, required)
        external = (env.piece_owner == 3) & (env.piece_pos[..., 0] < self.home_x)
        write_rows(self.external, external)
        write_rows(self.nearest_defender, torch.full_like(self.nearest_defender,
            math.hypot(env.field_length, env.sim.field_width)))
        write_rows(self.previous_x, env.sim.pose[:, 3, 0])

    @property
    def home_x(self):
        return self.env.field_length - self.env.alliance_zone_depth

    def acquired(self, selected, owners, positions):
        own = selected & (owners == 3)
        credited = own & ~self.acquisition_credit
        count = credited.sum(-1).float()
        self.acquisitions += count
        self.delta += .02 * count
        self.acquisition_credit |= own
        self.external |= own & (positions[..., 0] < self.home_x)
        # Recollection removes physical stock and pending landing, but preserves
        # delivery credit until a genuine score starts a new piece lifecycle.
        self.pending &= ~selected
        self.settling.masked_fill_(selected, 0.)

    def _delivery_batch(self, own):
        starting = own.any(-1) & ~self.delivery_open
        self.batch_id += starting.long()
        self.delivery_open |= starting
        return self.batch_id[:, None].expand_as(own)

    def released(self, selected, owners):
        own = selected & (owners == 3)
        batch = self._delivery_batch(own)
        self.piece_batch.copy_(torch.where(own, batch, self.piece_batch))
        self.pending |= own & self.external & ~self.ferry_credit
        self.inactive_origin |= own & ~self.env.hub_active[:, 1, None]
        self.settling.masked_fill_(own, 0.)

    def scored(self, selected, owners):
        own = selected & (owners == 3)
        teammate = selected & (owners >= 4) & self.ferry_credit
        self.team_conversions += teammate.sum(-1).float()
        converted = selected & (owners >= 3) & self.ferry_credit
        self.staged_conversions += converted.sum(-1).float()
        self.initial_staged_conversions += (converted & self.initial_staged).sum(-1).float()
        self.inactive_conversions += (converted & self.inactive_origin).sum(-1).float()
        self.delta += 2. * teammate.sum(-1).float()
        batch = self._delivery_batch(own)
        count = own.sum(-1).float()
        self.scores += count
        schedule = self.env.curriculum["schedule"]
        boundary = schedule[1][0] if schedule and len(schedule) > 1 else float("inf")
        self.post_boundary_scores += count * (self.env.match_elapsed >= boundary).float()
        self.delta += 2. * count
        self.batch_counts.scatter_add_(1, batch, own.float())
        self.score_batch_counts.scatter_add_(1, batch, own.float())
        self.score_batch_counts.scatter_add_(1, self.piece_batch, teammate.float())
        self.lifecycle += selected.long()
        for tensor in (self.acquisition_credit, self.external, self.ferry_credit, self.pending, self.initial_staged, self.inactive_origin):
            tensor.masked_fill_(selected, False)
        self.settling.masked_fill_(selected, 0.)

    def update(self, active):
        env = self.env
        self.elapsed += active.float() * env.dt
        friendly = env.hub_active[:, 1]
        self.active_seconds += (active & friendly).float() * env.dt
        self.inactive_seconds += (active & ~friendly).float() * env.dt
        free = env.piece_active & (env.piece_owner < 0)
        at_home = ((env.piece_pos[..., 0] >= self.home_x)
                   & (env.piece_pos[..., 0] <= env.field_length - .15)
                   & (env.piece_pos[..., 1] >= .15)
                   & (env.piece_pos[..., 1] <= env.sim.field_width - .15))
        if env.fuel_physics is None:
            settled = env.piece_vel.norm(dim=-1) < .3
        else:
            settled = ((env.fuel_physics.pos[..., 2] <= env.fuel_physics.config.radius + .03)
                       & (env.fuel_physics.vel.norm(dim=-1) < .3))
        stable = free & at_home & settled & self.pending
        self.settling.copy_(torch.where(active[:, None],
            torch.where(stable, self.settling + env.dt, 0.), self.settling))
        delivered = stable & (self.settling >= .5) & ~self.ferry_credit & active[:, None]
        count = delivered.sum(-1).float()
        self.deliveries += count
        self.inactive_deliveries += (delivered & self.inactive_origin).sum(-1).float()
        schedule = env.curriculum["schedule"]
        boundary = schedule[1][0] if schedule and len(schedule) > 1 else float("inf")
        self.post_boundary_deliveries += count * (env.match_elapsed >= boundary).float()
        self.delta += count
        self.ferry_credit |= delivered
        self.pending &= ~delivered
        self.batch_counts.scatter_add_(1, self.piece_batch, delivered.float())
        self.ferry_batch_counts.scatter_add_(1, self.piece_batch, (delivered & self.inactive_origin).float())
        stock = free & at_home & settled & self.ferry_credit
        self.current_staged = stock.sum(-1).float()
        self.inactive_staged = (stock & self.inactive_origin).sum(-1).float()
        empty = (env.piece_owner == 3).sum(-1) == 0
        ended = empty & self.delivery_open & active
        self.batch_ended.scatter_(1, self.batch_id[:, None],
            self.batch_ended.gather(1, self.batch_id[:, None]) | ended[:, None])
        self.delivery_open &= ~ended
        self.batch_completed |= self.batch_ended & (self.batch_counts >= env.curriculum['targets']['batch_size'])
        slots = env.curriculum["pressure"]["slots"]
        if slots:
            distances = (env.sim.pose[:, slots, :2] - env.sim.pose[:, 3, None, :2]).norm(dim=-1).amin(-1)
            self.nearest_defender = torch.where(active, torch.minimum(self.nearest_defender, distances), self.nearest_defender)
            self.encounter_ticks += ((distances <= 2.5) & active).float()
        route = env.curriculum['route']
        x, y = env.sim.pose[:, 3, 0], env.sim.pose[:, 3, 1]
        boundary = float(env.hub_centers[1, 0])
        crossed = ((self.previous_x < boundary) & (x >= boundary)) | ((self.previous_x > boundary) & (x <= boundary))
        if route == 'trench':
            crossed &= (y < 1.27) | (y > env.sim.field_width - 1.27)
        elif route == 'bump':
            crossed &= ((y > 1.67) & (y < 3.44)) | ((y > 4.63) & (y < 6.40))
        self.route_seen |= crossed & active
        self.previous_x.copy_(torch.where(active, x, self.previous_x))

    def outcomes(self):
        env = self.env
        name, targets = env.curriculum['scenario_id'], env.curriculum['targets']
        held = (env.piece_owner == 3) & env.piece_active & self.required
        retained = held.sum(-1).float()
        cycles = self.batch_completed.sum(-1)
        score_cycles = (self.batch_ended & (self.score_batch_counts >= targets['batch_size'])).sum(-1)
        ferry_cycles = (self.batch_ended & (self.ferry_batch_counts >= targets['batch_size'])).sum(-1)
        useful = self.scores + self.team_conversions + self.current_staged
        active_category = name.startswith('active_') or env.curriculum['category'].startswith('active/')
        if name in ('active_collect_carry_score', 'inactive_collect_ferry', 'offense_vs_scripted_pressure', 'team_delivery_cycles', 'full_match_offense'):
            if active_category:
                useful = self.scores + self.team_conversions
                cycles = score_cycles
            else:
                useful = self.inactive_conversions + self.inactive_staged
                cycles = ferry_cycles
        success = (self.scores >= targets['amount'])
        if name.startswith('pickup_'):
            success = retained >= targets['amount']
            useful = retained
        elif name == 'active_carry_to_zone':
            success = ((retained >= targets['amount']) & env.hub_active[:, 1]
                       & (env.sim.pose[:, 3, 0] >= self.home_x)
                       & env._robots_clear_of_bumps(env.sim.pose[:, :, :2])[:, 3]
                       & self.route_seen)
        elif name == 'inactive_ferry_preloaded':
            success = self.current_staged >= targets['amount']
        elif name in ('active_collect_carry_score', 'inactive_collect_ferry'):
            success = (cycles >= 2) & (useful >= targets['amount'])
        elif name in ('hub_transition_cycles', 'offense_vs_scripted_pressure', 'team_delivery_cycles', 'full_match_offense'):
            success = (cycles >= targets['cycles']) & (useful >= targets['amount'])
            if name == 'hub_transition_cycles':
                success &= (self.active_seconds > 0) & (self.inactive_seconds > 0)
                if env.curriculum['category'].startswith('active_to_inactive'):
                    success &= (self.post_boundary_deliveries >= 1) | ((retained >= 1) & ~env.hub_active[:, 1])
                else:
                    success &= self.post_boundary_scores >= 1
                if env.curriculum['category'].endswith('staged_at_home'):
                    success &= self.initial_staged_conversions >= 1
            if name == 'full_match_offense':
                if env.curriculum['category'].startswith('inactive/'):
                    useful = self.inactive_conversions + self.inactive_staged
                    success &= (self.inactive_seconds > 0) & (useful >= targets['amount'])
                else:
                    useful = self.scores
                    success &= self.scores >= targets['amount']
                success &= env.match_elapsed >= 160. - env.dt/2
            if env.curriculum['pressure']['level'] > 0:
                success &= self.encounter_ticks >= 5
            if name == 'team_delivery_cycles' and env.curriculum['category'].startswith('inactive/'):
                success &= self.team_conversions >= 1
        if env.curriculum['route'] in ('trench', 'bump') and name in ('active_collect_carry_score', 'inactive_collect_ferry', 'offense_vs_scripted_pressure', 'team_delivery_cycles', 'full_match_offense'):
            success &= self.route_seen
        success &= self.elapsed <= env.curriculum['time_budget_s'] + env.dt
        if name == 'active_carry_to_zone':
            useful = torch.where(success, retained, 0.)
        return {'success': success, 'useful_throughput': useful / self.elapsed.clamp_min(env.dt),
                'distinct_retained': retained, 'current_staged': self.current_staged.clone(),
                'completed_cycles': cycles, 'score_cycles': score_cycles, 'ferry_cycles': ferry_cycles,
                'inactive_staged': self.inactive_staged.clone(), 'scores': self.scores.clone(),
                'acquisitions': self.acquisitions.clone(), 'deliveries': self.deliveries.clone(),
                'route_seen': self.route_seen.clone(), 'elapsed': self.elapsed.clone(),
                'team_conversions': self.team_conversions.clone(),
                'staged_conversions': self.staged_conversions.clone(),
                'initial_staged_conversions': self.initial_staged_conversions.clone(),
                'inactive_deliveries': self.inactive_deliveries.clone(),
                'active_seconds': self.active_seconds.clone(),
                'inactive_seconds': self.inactive_seconds.clone(),
                'post_boundary_scores': self.post_boundary_scores.clone(),
                'post_boundary_deliveries': self.post_boundary_deliveries.clone(),
                'encounter_ticks': self.encounter_ticks.clone(),
                'nearest_defender': self.nearest_defender.clone()}

    def reward_delta(self):
        """Consume the accumulated per-tick increments exactly once."""
        result = self.delta.clone()
        self.delta.zero_()
        return result
