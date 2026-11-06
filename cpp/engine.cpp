#include "engine.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace qbt {

Engine::Engine(int T, int N, std::vector<double> open, std::vector<double> high, std::vector<double> low,
               std::vector<double> close, std::vector<double> volume, Config cfg)
    : T_(T), N_(N), open_(std::move(open)), high_(std::move(high)), low_(std::move(low)), close_(std::move(close)),
      volume_(std::move(volume)), cfg_(cfg) {
  const auto n = static_cast<std::size_t>(T) * N;
  for (auto* v : {&open_, &high_, &low_, &close_, &volume_})
    if (v->size() != n) throw std::invalid_argument("price arrays must all be T x N");
  if (cfg_.latency_bars < 1) throw std::invalid_argument("latency_bars must be >= 1 (no look-ahead)");
}

void Engine::reset() {
  queue_ = {};
  orders_.clear();
  fills_.clear();
  pos_.assign(N_, 0.0);
  volume_used_.assign(N_, 0.0);
  cash_ = equity_ = cfg_.initial_cash;
  next_id_ = 1;
  seq_ = events_ = 0;
  equity_curve_.assign(T_, 0.0);
  cash_curve_.assign(T_, 0.0);
  pos_hist_.assign(static_cast<std::size_t>(T_) * N_, 0.0);
  for (int t = 0; t < T_; ++t)
    for (Phase p : {OPEN, INTRABAR, CLOSE}) queue_.push({t, p, seq_++});
}

int64_t Engine::submit(int sym, double qty, std::optional<double> limit_px, int tif_bars) {
  if (sym < 0 || sym >= N_) throw std::out_of_range("bad symbol index");
  if (qty == 0 || !std::isfinite(qty)) return 0;
  Order o{next_id_++, sym, qty, limit_px ? OrderType::Limit : OrderType::Market, limit_px.value_or(0.0),
          t_ + cfg_.latency_bars, INT_MAX};
  if (tif_bars > 0) o.t_expire = o.t_active + tif_bars - 1;
  orders_.push_back(o);
  return o.id;
}

void Engine::cancel(int64_t id) {
  for (auto& o : orders_)
    if (o.id == id) o.qty = 0;
}

void Engine::cancel_all() {
  for (auto& o : orders_) o.qty = 0;
}

double Engine::pending_qty(int sym) const {
  double q = 0;
  for (const auto& o : orders_)
    if (o.sym == sym) q += o.qty;
  return q;
}

void Engine::purge() {
  orders_.erase(std::remove_if(orders_.begin(), orders_.end(),
                               [this](const Order& o) { return o.qty == 0 || o.t_expire < t_; }),
                orders_.end());
}

void Engine::execute(Order& o, int t, double ref_px, bool aggressive) {
  const auto i = idx(t, o.sym);
  const double cap = std::max(0.0, cfg_.max_participation * volume_[i] - volume_used_[o.sym]);
  const double q = std::copysign(std::min(std::abs(o.qty), cap), o.qty);
  if (q == 0) return;
  const double side = q > 0 ? 1.0 : -1.0;
  double px = ref_px;
  if (aggressive) {
    const double impact = cfg_.impact_coef * std::sqrt(std::abs(q) / std::max(volume_[i], 1.0));
    px *= 1.0 + side * (cfg_.half_spread_bps * 1e-4 + impact);
  }
  const double commission = std::abs(q) * px * cfg_.commission_bps * 1e-4;
  pos_[o.sym] += q;
  cash_ -= q * px + commission;
  volume_used_[o.sym] += std::abs(q);
  o.qty -= q;
  fills_.push_back({t, o.sym, o.id, q, px, commission});
}

void Engine::on_open(int t) {
  std::fill(volume_used_.begin(), volume_used_.end(), 0.0);
  for (auto& o : orders_) {
    if (o.qty == 0 || o.t_active > t) continue;
    const double op = open_[idx(t, o.sym)];
    if (o.type == OrderType::Market) {
      execute(o, t, op, true);
    } else if ((o.qty > 0 && op <= o.limit_px) || (o.qty < 0 && op >= o.limit_px)) {
      execute(o, t, op, false);  // limit marketable at the open: price improvement to the open
    }
  }
}

void Engine::on_intrabar(int t) {
  for (auto& o : orders_) {
    if (o.qty == 0 || o.t_active > t || o.type != OrderType::Limit) continue;
    const auto i = idx(t, o.sym);
    if ((o.qty > 0 && low_[i] < o.limit_px) || (o.qty < 0 && high_[i] > o.limit_px))
      execute(o, t, o.limit_px, false);  // strict cross: touching the price isn't enough to get filled
  }
}

void Engine::on_close(int t, const Strategy& strategy) {
  double mtm = 0;
  for (int s = 0; s < N_; ++s) {
    mtm += pos_[s] * close_[idx(t, s)];
    pos_hist_[idx(t, s)] = pos_[s];
  }
  equity_ = cash_ + mtm;
  equity_curve_[t] = equity_;
  cash_curve_[t] = cash_;
  purge();
  if (t + cfg_.latency_bars < T_) strategy(*this, t);
}

void Engine::run(const Strategy& strategy) {
  reset();
  while (!queue_.empty()) {
    const Event e = queue_.top();
    queue_.pop();
    ++events_;
    t_ = e.t;
    switch (e.phase) {
      case OPEN: on_open(e.t); break;
      case INTRABAR: on_intrabar(e.t); break;
      case CLOSE: on_close(e.t, strategy); break;
    }
  }
}

void Engine::run_targets(const std::vector<double>& w) {
  if (w.size() != static_cast<std::size_t>(T_) * N_) throw std::invalid_argument("weights must be T x N");
  run([this, &w](Engine& eng, int t) {
    // Replace any unfilled rebalance orders with a fresh order toward today's target.
    eng.cancel_all();
    eng.purge();
    for (int s = 0; s < N_; ++s) {
      const double wt = w[idx(t, s)];
      if (!std::isfinite(wt)) continue;
      const double px = close_[idx(t, s)];
      const double delta = wt * equity_ / px - pos_[s];
      if (std::abs(delta * px) > std::max(cfg_.min_trade_value, 1e-9)) eng.submit(s, delta);
    }
  });
}

}  // namespace qbt
