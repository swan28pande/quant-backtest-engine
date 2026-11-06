#pragma once
// Event-driven execution simulator.
//
// Each bar t produces three event phases, processed from a time-ordered queue:
//   OPEN      orders whose latency has elapsed become active; market orders fill at the open
//   INTRABAR  resting limit orders fill if the bar's range crosses their price
//   CLOSE     mark-to-market, record state, then run the strategy (which may submit orders)
// Fills are capped at a fraction of bar volume (partial fills carry over), and pay
// half-spread + square-root market impact (market orders) plus commission (all orders).
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <vector>

namespace qbt {

struct Config {
  double initial_cash = 1'000'000.0;
  double commission_bps = 0.5;
  double half_spread_bps = 1.0;
  double impact_coef = 0.1;        // impact (fraction of price) = coef * sqrt(fill / bar volume)
  double max_participation = 0.1;  // max fraction of bar volume per order per bar
  int latency_bars = 1;            // orders submitted at close t become active at open t+latency
  double min_trade_value = 0.0;    // target mode: skip rebalances smaller than this notional
};

enum class OrderType : uint8_t { Market, Limit };

struct Order {
  int64_t id;
  int sym;
  double qty;        // signed; remaining quantity
  OrderType type;
  double limit_px;
  int t_active;
  int t_expire;      // last bar the order can trade (inclusive); INT32_MAX = GTC
};

struct Fill {
  int t;
  int sym;
  int64_t order_id;
  double qty;
  double price;
  double commission;
};

class Engine {
 public:
  using Strategy = std::function<void(Engine&, int)>;

  // Prices are row-major (T x N) arrays.
  Engine(int T, int N, std::vector<double> open, std::vector<double> high, std::vector<double> low,
         std::vector<double> close, std::vector<double> volume, Config cfg);

  // Strategy API (valid inside the callback)
  int64_t submit(int sym, double qty, std::optional<double> limit_px = std::nullopt, int tif_bars = 0);
  void cancel(int64_t order_id);
  void cancel_all();
  double position(int sym) const { return pos_[sym]; }
  const std::vector<double>& positions() const { return pos_; }
  double cash() const { return cash_; }
  double equity() const { return equity_; }
  double pending_qty(int sym) const;
  int now() const { return t_; }
  const Config& config() const { return cfg_; }
  double close_px(int t, int sym) const { return close_[idx(t, sym)]; }

  // Drivers
  void run(const Strategy& strategy);
  void run_targets(const std::vector<double>& weights);  // (T x N) target portfolio weights

  // Results
  int T() const { return T_; }
  int N() const { return N_; }
  const std::vector<double>& equity_curve() const { return equity_curve_; }
  const std::vector<double>& cash_curve() const { return cash_curve_; }
  const std::vector<double>& position_history() const { return pos_hist_; }
  const std::vector<Fill>& fills() const { return fills_; }
  uint64_t events_processed() const { return events_; }

 private:
  enum Phase : uint8_t { OPEN = 0, INTRABAR = 1, CLOSE = 2 };
  struct Event {
    int t;
    Phase phase;
    uint64_t seq;
    bool operator>(const Event& o) const {
      if (t != o.t) return t > o.t;
      if (phase != o.phase) return phase > o.phase;
      return seq > o.seq;
    }
  };

  std::size_t idx(int t, int s) const { return static_cast<std::size_t>(t) * N_ + s; }
  void reset();
  void on_open(int t);
  void on_intrabar(int t);
  void on_close(int t, const Strategy& strategy);
  void execute(Order& o, int t, double px, bool aggressive);
  void purge();

  int T_, N_;
  std::vector<double> open_, high_, low_, close_, volume_;
  Config cfg_;

  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> queue_;
  std::vector<Order> orders_;
  std::vector<double> pos_, volume_used_;
  double cash_ = 0, equity_ = 0;
  int t_ = 0;
  int64_t next_id_ = 1;
  uint64_t seq_ = 0, events_ = 0;

  std::vector<double> equity_curve_, cash_curve_, pos_hist_;
  std::vector<Fill> fills_;
};

}  // namespace qbt
